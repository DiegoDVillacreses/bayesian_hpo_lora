import os
import pickle

import torch
import shutil

import gc

from tqdm import tqdm

import optuna

import pandas as pd
import numpy as np
import random
from torch import nn
from transformers import AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint,Callback, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torchmetrics.classification import Accuracy, F1Score
from torchmetrics import FBetaScore
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification,get_linear_schedule_with_warmup
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    classification_report,
    fbeta_score
)
from torchmetrics import Metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ── 0) User defined functions ────────────────────────────────────────
class CustomFBeta(Metric):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta
        # create state variables
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        # convert logits → binary predictions
        #preds = (torch.sigmoid(preds) > 0.5).int()
        self.tp += torch.sum((preds == 1) & (target == 1))
        self.fp += torch.sum((preds == 1) & (target == 0))
        self.fn += torch.sum((preds == 0) & (target == 1))

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        b2 = self.beta ** 2
        return (1 + b2) * (precision * recall) / (b2 * precision + recall + 1e-8)
    
def seed_everything_everywhere(seed: int = 0):
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch (CPU)
    torch.manual_seed(seed)
    # PyTorch (all GPUs)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN determinism (may slow you down, but ensures reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # PyTorch Lightning’s helper (also seeds Python, NumPy, torch)
    pl.seed_everything(seed, workers=True)

def clear_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ── 1) Load data ────────────────────────────────────────
model_name = 'meta-llama/Llama-3.2-3B-Instruct'
df = pd.read_csv("train_text.csv")

train_df = df.sample(frac=0.7, random_state=1)
val_df   = df.drop(train_df.index)


# ── 2) LIGHTNING MODULEs ───────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            # **no return_tensors here!**
        )
        # enc["input_ids"] is now a list[int], same for attention_mask
        return {
            "input_ids":     enc["input_ids"],
            "attention_mask":enc["attention_mask"],
            "labels":        self.labels[idx]
        }

class TextClassificationDataModule(pl.LightningDataModule):
    def __init__(
            self, train_dataset, 
            val_dataset, 
            batch_size: int = 8, 
            num_workers: int = 4, 
            collate_fn=None
            ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=torch.cuda.is_available()
        )
    
class LitLlama(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        class_weights: torch.Tensor,
        lr: float = 3e-4,
        r: int = 8,
        beta: float = 1.0,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        weight_decay = 0.01,
        last_n = 6,
        warmup_epochs: int = 0
    ):
        super().__init__()
        # only save simple hyperparameters; ignore non-serializable tensors
        self.save_hyperparameters(ignore=['class_weights'])

        # Tokenizer: use AutoTokenizer for Llama 3
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )  # correct tokenizer for Llama 3 :contentReference[oaicite:4]{index=4}

        # Base model: sequence classification head on Llama 3
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            trust_remote_code=True
        )  # picks up LlamaForSequenceClassification :contentReference[oaicite:5]{index=5}
        base_model.config.pad_token_id = self.tokenizer.eos_token_id
        # Apply LoRA to query and value projections
        total_layers = base_model.config.num_hidden_layers       # 24 for Llama-3 3B
        layers_to_transform = list(range(total_layers-last_n, total_layers))

        peft_cfg = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
                r=r,
                lora_alpha=32,
                lora_dropout=lora_dropout,
                # LlamaAttention has q_proj, k_proj, v_proj and o_proj
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                layers_to_transform=layers_to_transform
                #modules_to_save=["classifier"]
            )
        self.model = get_peft_model(base_model, peft_cfg)

        # Loss
        #self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_labels)
        self.train_f1_macro  = F1Score(task="multiclass", num_classes=num_labels, average="macro")
        self.val_f1_macro    = F1Score(task="multiclass", num_classes=num_labels, average="macro")
        self.train_f1_weight = F1Score(task="multiclass", num_classes=num_labels, average="weighted")
        self.val_f1_weight   = F1Score(task="multiclass", num_classes=num_labels, average="weighted")
        # self.train_fbeta = FBetaScore(task='binary', num_classes=num_labels, beta=beta)
        # self.val_fbeta   = FBetaScore(task='binary', num_classes=num_labels, beta=beta)
        #self.train_fbeta = FBetaScore(task='multiclass', num_classes=num_labels, beta=beta, average='macro')
        #self.val_fbeta   = FBetaScore(task='multiclass', num_classes=num_labels, beta=beta, average='macro')
        self.train_fbeta = CustomFBeta(beta=beta)
        self.val_fbeta   = CustomFBeta(beta=beta)        
        
        #FBetaScore(task='multiclass', num_classes=2, beta=3.0, average='macro')

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        labels = batch["labels"]
        outputs = self.model(**batch)
        loss = F.cross_entropy(outputs.logits, labels, weight=self.class_weights)
        preds = torch.argmax(outputs.logits, dim=1)

        self.log_dict(
            {
                "train_loss": loss,
                "train_acc":  self.train_acc(preds, labels),
                "train_f1_macro":  self.train_f1_macro(preds, labels),
                #"train_f1_weight": self.train_f1_weight(preds, labels),
                #"train_fbeta":     self.train_fbeta(preds, labels),
                "train_fbeta":     self.train_fbeta(preds,labels),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        This function is now only responsible for the forward pass,
        calculating the loss, and UPDATING the state of the metrics.
        """
        labels = batch["labels"]
        logits = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        ).logits

        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        preds = torch.argmax(logits, dim=1)

        # Update the state of each metric. This accumulates the TP, FP, etc.
        # for each batch without computing the final score yet.
        self.val_acc.update(preds, labels)
        self.val_f1_macro.update(preds, labels)
        self.val_fbeta.update(preds, labels)

        # We log the loss on an epoch-wide basis. Lightning will average it.
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self):
        """
        This new hook is called once at the very end of the validation epoch.
        It's the correct place to COMPUTE and LOG the final epoch-wide metrics.
        """
        # Compute the final score from the accumulated state of all batches
        final_acc = self.val_acc.compute()
        final_f1 = self.val_f1_macro.compute()
        final_fbeta = self.val_fbeta.compute()

        # Log the correct, epoch-wide metrics
        self.log_dict({
            "val_acc": final_acc,
            "val_f1_macro": final_f1,
            "val_fbeta": final_fbeta,
        }, prog_bar=True)

        # PyTorch Lightning automatically calls .reset() on the metrics
        # after this hook, so they are ready for the next epoch.

    # -------------------- CHANGE ENDS HERE --------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.hparams.warmup_epochs * self.trainer.num_training_batches

        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    

# ── 3) TRAINING Function ───────────────────────────────────────────────────
def lora_training(
        weight_decay, 
        last_n, 
        beta, 
        lr, 
        r, 
        weights, 
        lora_dropout, 
        patience = 10, 
        warmup_epochs = 3, 
        max_epochs = 200, 
        num_workers = 4, 
        model_max_length = 512
        ):
    folder_paths = [
        'checkpoints',
        'logs'
    ]
    for folder_path in folder_paths:
        # Check if folder exists
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)
            print(f"Folder '{folder_path}' deleted.")
        else:
            print(f"Folder '{folder_path}' does not exist.")
    checkpoint_cb = ModelCheckpoint(
        monitor="val_fbeta",
        mode="max",
        save_top_k=1,
        dirpath="checkpoints/",
        filename="best-{epoch:02d}-{val_loss:.2f}",
        save_weights_only=True
    )
    earlystop_cb = EarlyStopping(
        monitor="val_fbeta",
        mode="max",
        patience=patience
        )

    model = LitLlama(
        model_name=model_name,
        lora_dropout = lora_dropout,
        num_labels=2,
        class_weights=weights,
        r = r,
        lr=lr,
        beta = beta,
        last_n = last_n,
        weight_decay = weight_decay,
        warmup_epochs = warmup_epochs
    )
    csv_logger = CSVLogger(save_dir='./logs', name='csv_logs')

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[earlystop_cb,checkpoint_cb,lr_monitor],
        logger=csv_logger,
        accelerator="auto",
        precision=16 if torch.cuda.is_available() else 32,
    )

    def collate_fn(batch):
        """
        batch: list of dicts that each contain
            {input_ids, attention_mask, (optional) token_type_ids, labels}
        returns: single dict with same keys, each value a tensor on CPU
        """
        # 1) pull out the labels before padding
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

        # 2) strip labels from the feature dicts
        features = [{k: v for k, v in item.items() if k != "labels"} for item in batch]

        # 3) pad the usual way
        batch_padded = tokenizer.pad(
            features,
            padding="longest",          # or 'max_length'
            return_tensors="pt"
        )

        # 4) put the labels back
        batch_padded["labels"] = labels
        return batch_padded

    seed_everything_everywhere(0)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
        model_max_length=model_max_length
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token    = tokenizer.eos_token

    train_dataset = TextDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length=model_max_length
    )
    val_dataset = TextDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length=model_max_length
    )
    dm = TextClassificationDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    trainer.fit(
        model = model,
        datamodule=dm
    )
    clear_cuda()
    res = trainer.validate(ckpt_path=checkpoint_cb.best_model_path, datamodule=dm)  
    log_dir = trainer.logger.log_dir
    try:
        metrics_df = pd.read_csv(
            os.path.join(log_dir, "metrics.csv"),
            engine="python",
            on_bad_lines="skip",       
            skip_blank_lines=True
        )

        train = (
            metrics_df[metrics_df["train_acc"].notna()]
            .groupby("epoch", as_index=True)
            .agg(train_acc = ("train_acc", "last"),
                train_loss= ("train_loss", "last"))
        )

        val = (
            metrics_df[metrics_df["val_acc"].notna()]
            .groupby("epoch", as_index=True)
            .agg(val_acc  = ("val_acc", "last"),
                val_loss = ("val_loss", "last"))
        )
        df_epoch = train.join(val)
        df_epoch.reset_index(inplace=True)
    except:
        df_epoch = None
    clear_cuda()
    return res, df_epoch


def objective(trial: optuna.trial.Trial):
    lr    =         trial.suggest_categorical("lr", [3e-6,5e-6,1e-5, 3e-5, 5e-5])
    weight_decay =  trial.suggest_categorical("weight_decay", [0,1e-3,3e-3,5e-3,3e-2,1e-2])
    dropout =       trial.suggest_categorical("dropout", [0, 0.05, 0.1, 0.15])
    r = trial.suggest_categorical("r", [16, 20, 24, 28])
    last_n = trial.suggest_categorical("last_n", [10, 12, 14])
    weight_loss = trial.suggest_categorical("weight_loss", [8.0, 9., 10., 11., 12., 13., 14., 15.])
    print("-"*50)
    print(f"Learning Rate: {lr}")
    print(f"weight_decay: {weight_decay}")
    print(f"dropout: {dropout}")


    lora_res = lora_training(
        weights = torch.tensor([1.0,weight_loss]),
        max_epochs = 200,
        num_workers = 4,
        beta = 3.0,
        patience = 15,
        lora_dropout = dropout,
        r = r,
        weight_decay = weight_decay,
        lr = lr,
        last_n = last_n,
        warmup_epochs = 3,
        model_max_length = 2048
    )
    f1_beta = lora_res[0][0]['val_fbeta']        
    print(f1_beta)
    return f1_beta

def make_no_improvement_callback(k: int):
    best_value = None
    best_trial_no = 0

    def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        nonlocal best_value, best_trial_no

        # we assume higher is better; if your objective is minimize, invert the comparison
        if best_value is None or (trial.value is not None and trial.value > best_value):
            best_value = trial.value
            best_trial_no = trial.number
        elif trial.number - best_trial_no >= k:
            # stop the study once we've had k consecutive non-improving trials
            study.stop()

    return callback

seed_everything_everywhere(0)

optuna_dir_full = '/home/dvillacreses/LoRA/optuna/'
try:
    os.makedirs(optuna_dir_full, exist_ok=True)
except:
    pass
try: 
    os.remove(os.path.join(optuna_dir_full,'my_study.db'))
except:
    pass

storage_path = f"sqlite:///{optuna_dir_full}/my_study.db"

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(),
    load_if_exists=True,
    storage=storage_path
)
patience_hpo = 25
stop_callback = make_no_improvement_callback(patience_hpo)
study.optimize(
    objective, 
    n_trials=200, 
    timeout=3600*24,
    callbacks=[stop_callback]
    )

with open("optuna_res.pkl", "wb") as f:  # 'wb' = write binary
    pickle.dump(study, f)