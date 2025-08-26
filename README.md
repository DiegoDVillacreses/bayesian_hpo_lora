# LoRA HPO — Clean, Reproducible Bayesian Optimization for LLM Fine-Tuning

> A minimal, research-grade template to **tune LoRA** adapters with **Bayesian Optimization** (Optuna/TPE), emphasizing **clarity, reproducibility, and auditability**.&#x20;

---

## Why this matters

* **Clean code ⇒ credible results.** Deterministic seeding, explicit data splits, and pinned metrics reduce “HPO noise” and make ablations trustworthy.
* **Bayesian Opt (TPE) ⇒ fewer, smarter trials.** You get strong configs for LoRA with far fewer evaluations than grid/random.
* **LoRA ⇒ parameter-efficient science.** Move fast on constrained GPUs while staying faithful to baselines and reporting.

---

## What’s inside

* **Task:** sequence classification on Llama-3.x backbone with **LoRA** adapters.
* **HPO knobs (via Optuna/TPE):** `lr`, `weight_decay`, `lora_dropout`, `r`, `last_n` layers, and **class-weight** for imbalance.
* **Metrics:** accuracy, macro-F1, and **Fβ (β=3)** for recall-sensitive tasks.
* **Reproducibility:** a single `seed_everything_everywhere(0)` seeds Python/NumPy/PyTorch/CUDA/Lightning; CuDNN locked for determinism.
* **Logging & artifacts:** CSV metrics per epoch, best checkpoint by `val_fbeta`, and a pickled Optuna study.


## Outputs you’ll get

* `logs/csv_logs/metrics.csv` — per-epoch training/validation metrics (for plots/tables).
* `checkpoints/best-*.ckpt` — best model by **val Fβ**.
* `optuna_res.pkl` — full Optuna study (trials, values, best params).

Load study later:

```python
import pickle
study = pickle.load(open("optuna_res.pkl","rb"))
print(study.best_trial.params, study.best_value)
```

---

## Design choices that keep results honest

* **Deterministic everything:** Python/NumPy/CPU/GPU seeds + CuDNN determinism + Lightning seeding.
* **Explicit collate & padding:** tokenizer-driven padding to avoid silent truncation drift across runs.
* **Early stopping & model selection:** monitor `val_fbeta` to align selection with the paper’s objective.
* **Clear search spaces:** learning rate, decay, dropout, rank `r`, final `last_n` LoRA layers, and class-weight all recorded by Optuna.
* **Cache hygiene:** GPU memory cleared and synchronized between phases to avoid OOM-induced variance.

---

## Minimal code map

* **`lora_hpo.py`** — complete pipeline: data split (70/30), Lightning module, LoRA injection, scheduler, metrics, Optuna study & callbacks.

  * Backbone: `meta-llama/Llama-3.2-3B-Instruct` (sequence classification head).
  * LoRA on attention proj. (`q,k,v,o`) with optional **last-N layer targeting**.
  * Imbalance-aware CE via tunable class weights.

---

## Reproduce & extend

* **Change the backbone:** swap `model_name` to any seq-classification-friendly LLM.
* **Different objective:** switch the metric monitored in `ModelCheckpoint` (e.g., macro-F1).
* **Larger contexts:** bump `model_max_length` (2048 shown) and ensure VRAM fit.
* **New search dims:** add priors (e.g., scheduler warmup, batch size) in `objective()`.

---

## Cite & background (suggested)

* Hu et al., 2022 — *LoRA: Low-Rank Adaptation of Large Language Models*.
* Bergstra & Bengio, 2012 — *Random Search for Hyper-Parameter Optimization*.
* Akiba et al., 2019 — *Optuna: A Next-generation Hyperparameter Optimization Framework*.
* Snoek, Larochelle & Adams, 2012 — *Practical Bayesian Optimization of Machine Learning Algorithms*.

---

## License & contribution

Use, adapt, and submit PRs that keep the code **small, legible, and testable**—that’s the whole point. If you publish with it, please reference this repo and the works above.

---

**TL;DR**: This repo is a **transparent scaffold** for LoRA fine-tuning where **Bayesian Optimization** is not an afterthought. Fork it to make your HPO **defensible** in a paper or report.&#x20;
