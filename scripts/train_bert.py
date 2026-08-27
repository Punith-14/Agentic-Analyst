"""Fine-tune a text critic on trajectory steps.

    python scripts/train_bert.py                          # uses text_config.json
    python scripts/train_bert.py --format full --max-len 256
    python scripts/train_bert.py --epochs 4 --lr 3e-5 --tag lr3e5
    python scripts/train_bert.py --smoke                  # 2 min, CPU, checks plumbing

WHY THIS IS A SCRIPT AND NOT A NOTEBOOK
Fine-tuning is 20-40 minutes on a 6 GB card. A notebook loses the run when the
kernel dies or the laptop sleeps, can't queue three configurations overnight,
and after an hour of cell-shuffling you can no longer say which state produced
the model sitting in memory. The command line above IS the record of what ran.

The input format is decided in notebooks/04_bert_inputs.ipynb and read from
data/critic/text_config.json, so the script and the notebook cannot silently
disagree about what the model sees. Both build strings through
agent/critic/text.py for the same reason.

THE TEST SET IS NOT TOUCHED HERE.
Early stopping needs a validation set, so one is carved out of the training
runs — grouped by run_id, same as the main split. Scoring against the frozen
test set happens once, in notebook 05. A validation set used for model
selection is no longer an honest estimate of anything.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.critic.text import FORMATS, build_text

DATA = ROOT / "data" / "critic"
MODELS = ROOT / "models"


# ==========================================================================
# environment guards — fail before the expensive part, not during it
# ==========================================================================

def ollama_is_holding_the_gpu() -> tuple[bool, str]:
    """Ollama keeps a model resident in VRAM for five minutes after a request.

    On a 6 GB card a 3B model plus a fine-tuning run does not fit, and the
    failure arrives as a CUDA OOM three minutes into epoch one that reads like
    a bug in this script. Cheaper to check up front.
    """
    if not shutil.which("ollama"):
        return False, "ollama not installed"
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True,
                             timeout=10).stdout
    except Exception as e:                                  # noqa: BLE001
        return False, f"could not run `ollama ps` ({type(e).__name__})"

    lines = [l for l in out.strip().splitlines()[1:] if l.strip()]
    if lines:
        return True, "\n".join("      " + l for l in lines)
    return False, "no model resident"


def report_gpu() -> float:
    """Returns free VRAM in GB, 0.0 if there's no usable GPU."""
    try:
        import torch
    except ImportError:
        print("  torch not installed — see requirements-ml.txt")
        return 0.0

    if not torch.cuda.is_available():
        print("  no CUDA device — training on CPU will take hours, not minutes")
        return 0.0

    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    free_gb, total_gb = free / 1e9, total / 1e9
    print(f"  gpu:  {name}")
    print(f"  vram: {free_gb:.1f} GB free of {total_gb:.1f} GB")
    return free_gb


def suggest_batch_size(free_gb: float, max_len: int) -> int:
    """Rough, deliberately conservative. An OOM at minute 30 costs more than a
    slightly slower run."""
    if free_gb <= 0:
        return 8
    budget = free_gb - 1.2                      # activations, optimiser, fragmentation
    per_sample = max_len / 256 * 0.22           # ~0.22 GB per sample at len 256
    return int(max(4, min(32, budget / max(per_sample, 0.05))))


# ==========================================================================
# data
# ==========================================================================

@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_split(fmt: str, val_frac: float = 0.15, seed: int = 23) -> Split:
    df = pd.read_parquet(DATA / "labelled_steps.parquet")
    split = json.loads((DATA / "split.json").read_text())

    df = df[df.y_run_fails.notna()].copy()
    df["y"] = df.y_run_fails.astype(int)
    df["text"] = [build_text(r, fmt) for _, r in df.iterrows()]

    test_runs = set(split["test_runs"])
    test = df[df.run_id.isin(test_runs)].copy()
    pool = df[~df.run_id.isin(test_runs)].copy()

    # Validation carved by run_id, never by step. Steps inside one run are
    # near-duplicates; splitting by step puts the answer on both sides.
    runs = np.array(sorted(pool.run_id.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(runs)
    n_val = int(len(runs) * val_frac)
    val_runs = set(runs[:n_val])

    train = pool[~pool.run_id.isin(val_runs)].copy()
    val = pool[pool.run_id.isin(val_runs)].copy()

    for name, a, b in [("train/val", train, val), ("train/test", train, test),
                       ("val/test", val, test)]:
        shared = set(a.run_id) & set(b.run_id)
        assert not shared, f"run_id leak in {name}: {len(shared)} runs"

    return Split(train, val, test)


# ==========================================================================
# metrics
# ==========================================================================

def compute_metrics(eval_pred):
    """PR-AUC, not accuracy.

    The base rate is ~79% failures, so a model that answers "fails" every time
    scores 79% accuracy and is worthless. PR-AUC is threshold-free and is what
    the feature critic was selected on, which keeps the two comparable.
    """
    from scipy.special import softmax
    from sklearn.metrics import average_precision_score, roc_auc_score

    logits, labels = eval_pred
    probs = softmax(logits, axis=1)[:, 1]
    return {
        "pr_auc": float(average_precision_score(labels, probs)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "base_rate": float(np.mean(labels)),
    }


# ==========================================================================
# training
# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="HF model id")
    ap.add_argument("--format", default=None, choices=FORMATS)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = pick from free VRAM")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model, 200 rows, 1 epoch — proves the plumbing on CPU")
    ap.add_argument("--force", action="store_true",
                    help="train even if Ollama is holding the GPU")
    args = ap.parse_args()

    # --- config, with the notebook's decision as the default ---------------
    cfg_path = DATA / "text_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    if not cfg:
        print(f"! {cfg_path.name} not found — run notebooks/04_bert_inputs.ipynb\n"
              f"  first, or pass --format and --max-len explicitly.\n")

    model_id = args.model or cfg.get("model") or "answerdotai/ModernBERT-base"
    fmt = args.format or cfg.get("format") or "full"
    max_len = args.max_len or cfg.get("max_len") or 256

    if args.smoke:
        # distilbert, not bert-tiny. bert-tiny predates fast tokenizers and
        # ships only a vocab.txt, so transformers tries to convert it and
        # demands sentencepiece — which has no wheel on some Python versions.
        # distilbert ships tokenizer.json, so there is nothing to convert.
        model_id, max_len, args.epochs = "distilbert-base-uncased", 64, 1

    print("=" * 66)
    print("  text critic — fine-tuning")
    print("=" * 66)
    print(f"  model:  {model_id}")
    print(f"  format: {fmt}   max_len: {max_len}")

    # --- guards ------------------------------------------------------------
    if not args.smoke:
        busy, detail = ollama_is_holding_the_gpu()
        if busy:
            print("\n! Ollama currently has a model resident in VRAM:")
            print(detail)
            print("\n  6 GB will not hold a 3B language model and a fine-tuning run.")
            print("  Free it with:   ollama stop <model>")
            print("  Then rerun. Use --force to train anyway (expect CUDA OOM).")
            if not args.force:
                sys.exit(1)

    free_gb = report_gpu()
    batch = args.batch_size or suggest_batch_size(free_gb, max_len)
    print(f"  batch:  {batch}" + ("" if args.batch_size else "  (chosen from free VRAM)"))

    # --- data --------------------------------------------------------------
    print("\n  loading data ...")
    s = load_split(fmt, seed=args.seed)
    if args.smoke:
        s = Split(s.train.head(200), s.val.head(80), s.test.head(80))

    print(f"  train {len(s.train):>6,} steps  {s.train.run_id.nunique():>5,} runs  "
          f"base rate {s.train.y.mean():.3f}")
    print(f"  val   {len(s.val):>6,} steps  {s.val.run_id.nunique():>5,} runs  "
          f"base rate {s.val.y.mean():.3f}")
    print(f"  test  {len(s.test):>6,} steps  {s.test.run_id.nunique():>5,} runs  "
          f"HELD BACK — scored once in notebook 05")

    # --- tokenise ----------------------------------------------------------
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, EarlyStoppingCallback,
                              Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except ValueError as e:
        # Models without a tokenizer.json need sentencepiece/tiktoken to
        # convert. Say so plainly rather than leaving a 30-line traceback
        # about "backend tokenizer" for someone to decode.
        if "backend tokenizer" in str(e):
            sys.exit(
                f"\n! {model_id} ships only a slow tokenizer and transformers "
                f"cannot convert it.\n"
                f"  Either:  pip install sentencepiece\n"
                f"  Or pick a model that ships tokenizer.json, e.g.\n"
                f"           --model distilbert-base-uncased\n"
            )
        raise

    def encode(batch_):
        return tok(batch_["text"], truncation=True, max_length=max_len)

    def to_ds(frame):
        return (Dataset.from_pandas(frame[["text", "y"]].rename(columns={"y": "labels"}),
                                    preserve_index=False)
                .map(encode, batched=True, remove_columns=["text"]))

    ds_train, ds_val = to_ds(s.train), to_ds(s.val)

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    # Smoke runs get their own directory. Otherwise a 3-second distilbert run
    # on 200 rows lands in the same folder as the real model, and notebook 05
    # — which picks the first critic_bert_* it finds — would silently compare
    # LightGBM against a toy.
    tag = args.tag or ("smoke" if args.smoke else fmt)
    out_dir = MODELS / f"critic_bert_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # TrainingArguments moves between transformers releases — v5 dropped
    # warmup_ratio, v4.46 renamed evaluation_strategy to eval_strategy. Build
    # the dict, then keep only what this installed version actually accepts,
    # so the script survives an upgrade instead of dying on an unknown kwarg.
    import inspect
    import math

    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)

    ta = {
        "output_dir": str(out_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": batch,
        "per_device_eval_batch_size": batch * 2,
        "learning_rate": args.lr,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "pr_auc",
        "greater_is_better": True,
        "save_total_limit": 2,            # checkpoints are ~600 MB each
        "logging_steps": 25,
        "seed": args.seed,
        "fp16": torch.cuda.is_available(),   # roughly halves VRAM on the 3050
        "report_to": [],
    }

    # 10% warmup, expressed whichever way this version understands.
    steps_per_epoch = math.ceil(len(ds_train) / batch)
    if "warmup_ratio" in accepted:
        ta["warmup_ratio"] = 0.1
    else:
        ta["warmup_steps"] = max(1, int(0.1 * steps_per_epoch * args.epochs))

    # eval each epoch — the flag was renamed in 4.46
    ta["eval_strategy" if "eval_strategy" in accepted else "evaluation_strategy"] = "epoch"

    dropped = sorted(set(ta) - accepted)
    if dropped:
        print(f"  note: transformers {__import__('transformers').__version__} "
              f"does not accept {dropped} — skipping")
    targs = TrainingArguments(**{k: v for k, v in ta.items() if k in accepted})

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds_train, eval_dataset=ds_val,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n  training — {args.epochs} epochs, lr {args.lr}\n")
    t0 = time.perf_counter()
    trainer.train()
    mins = (time.perf_counter() - t0) / 60

    val_metrics = trainer.evaluate()
    print(f"\n  done in {mins:.1f} min")
    print(f"  val PR-AUC  {val_metrics['eval_pr_auc']:.4f}   "
          f"(base rate {val_metrics['eval_base_rate']:.4f})")

    # --- save --------------------------------------------------------------
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))

    meta = {
        "model_id": model_id, "format": fmt, "max_len": max_len,
        "batch_size": batch, "lr": args.lr, "epochs": args.epochs,
        "seed": args.seed,
        "train_steps": len(s.train), "val_steps": len(s.val),
        "train_runs": int(s.train.run_id.nunique()),
        "val_pr_auc": round(float(val_metrics["eval_pr_auc"]), 4),
        "val_roc_auc": round(float(val_metrics["eval_roc_auc"]), 4),
        "val_base_rate": round(float(val_metrics["eval_base_rate"]), 4),
        "minutes": round(mins, 1),
        "note": "test set NOT scored here — see notebooks/05_critic_comparison.ipynb",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)   # ~1.2 GB of nothing

    print(f"\n  -> {out_dir}")
    print("\n  Next: notebooks/05_critic_comparison.ipynb scores this against the")
    print("        frozen test set and compares it with the LightGBM critic.")
    print("=" * 66)


if __name__ == "__main__":
    main()
