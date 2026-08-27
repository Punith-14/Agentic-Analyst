"""Scores a live trajectory step so the loop can stop a doomed run early.

    from agent.critic.infer import CriticScorer
    critic = CriticScorer.load()
    p_fail = critic.score(question, steps)      # steps so far, newest last

Returns P(this run ends with a wrong answer), from the models trained in
notebooks 02 and 05:

    0.7 * LightGBM(25 numeric features)  +  0.3 * ModernBERT(step text)

Measured on 1,030 held-out steps: PR-AUC 0.9845, against 0.9819 for LightGBM
alone. The two disagree more than they agree about *which* steps are bad
(r = 0.735), which is why both are here — the text model catches queries that
run cleanly and answer the wrong question, and no count of joins can see that.

THE THING THAT WILL BREAK THIS
Train/serve skew. The models were fitted on rows produced by
scripts/label_dataset.py from logged JSONL. Here the rows come from live
TrajectoryStep objects. If those two paths disagree about even one column, the
model gets garbage in a shifted column and predicts confidently wrong — with no
error anywhere. _step_to_row() below mirrors label_dataset.flatten() field for
field, and tests/test_critic_infer.py checks the two paths agree numerically on
real logged steps. Change one, change the other.

NEVER RAISES. A critic that crashes the agent is worse than no critic, so every
public method swallows its own failures and falls back to "don't stop".
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from agent.critic.tabular import engineer
from agent.critic.text import build_text
from contracts import TrajectoryStep

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS = ROOT / "models"

# Chosen in notebook 02: the smallest threshold giving 95% precision on the
# "stop" decision. Stopping a healthy run destroys an answer; letting a doomed
# one finish wastes a few seconds. The two errors are not equally expensive.
DEFAULT_THRESHOLD = 0.694

# From notebook 05. The text model is the minority voice deliberately: it is
# less accurate alone (0.9741 vs 0.9819) but sees a failure class the feature
# model cannot.
W_LGBM, W_BERT = 0.7, 0.3


def _step_to_row(question: str, step: TrajectoryStep) -> dict:
    """One TrajectoryStep -> the columns label_dataset.flatten() produces.

    Only the fields engineer() and build_text() actually read. Kept in the
    same order as flatten() so the two are diffable by eye.
    """
    a = step.action
    obs = step.observation
    return {
        "step_index": step.step_index,
        "question": question,
        "thought": step.thought,
        "tool": a.tool if a else None,
        "sql": (a.args.get("query", "") if a and a.tool == "run_sql" else ""),
        "status": step.status,
        "obs_rows": obs.row_count if obs else None,
        "obs_truncated": obs.truncated if obs else False,
        "obs_error": obs.error if obs else None,
        "error_category": step.error_category,
        "consecutive_errors": step.consecutive_errors,
        "total_errors_so_far": step.total_errors_so_far,
        "parse_repair_count": step.parse_repair_count,
        "schema_inspected_before": step.schema_inspected_before,
        "tokens_in_prompt": step.tokens_in_prompt,
        "duration_ms": step.duration_ms,
    }


@dataclass
class CriticScorer:
    lgbm: object = None
    bert: object = None
    tokenizer: object = None
    device: str = "cpu"
    text_format: str = "full_err"
    max_len: int = 256
    w_lgbm: float = W_LGBM
    w_bert: float = W_BERT

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, lgbm_path: Path | None = None, bert_dir: Path | None = None,
             use_text: bool = True, device: str | None = None) -> "CriticScorer":
        """Load whatever is available. Missing models degrade, they don't raise.

        use_text=False skips the 600 MB text model entirely — worth it when the
        GPU is already hosting the language model, which on a 6 GB card it is.
        """
        import joblib

        self = cls()

        lgbm_path = lgbm_path or (MODELS / "critic_lgbm_v1.joblib")
        if lgbm_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")     # sklearn version chatter
                    self.lgbm = joblib.load(lgbm_path)
            except Exception as e:                              # noqa: BLE001
                # Unpickling an LGBMClassifier needs lightgbm importable, and
                # it is in requirements-ml.txt rather than requirements.txt —
                # so CI, which installs only the latter, gets here. Degrade
                # rather than raise: layer D imports this module and an
                # exception would take the orchestration tests down.
                print(f"  critic: feature model unavailable "
                      f"({type(e).__name__}: {e})")

        if use_text:
            bert_dir = bert_dir or self._best_bert_dir()
            if bert_dir and bert_dir.exists():
                try:
                    import json

                    import torch
                    from transformers import (AutoModelForSequenceClassification,
                                              AutoTokenizer)

                    meta_path = bert_dir / "meta.json"
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text())
                        self.text_format = meta.get("format", self.text_format)
                        self.max_len = meta.get("max_len", self.max_len)

                    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
                    self.tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
                    self.bert = (AutoModelForSequenceClassification
                                 .from_pretrained(str(bert_dir))
                                 .to(self.device).eval())
                except Exception as e:                          # noqa: BLE001
                    print(f"  critic: text model unavailable ({type(e).__name__}: {e}); "
                          f"falling back to features only")

        # Reweight rather than silently under-scoring if one model is absent.
        if self.lgbm is None and self.bert is None:
            raise FileNotFoundError(
                f"no critic models found in {MODELS}. Train one first:\n"
                f"  notebooks/02_critic_models.ipynb  (LightGBM)\n"
                f"  python scripts/train_bert.py      (text)")
        if self.bert is None:
            self.w_lgbm, self.w_bert = 1.0, 0.0
        elif self.lgbm is None:
            self.w_lgbm, self.w_bert = 0.0, 1.0

        return self

    @staticmethod
    def _best_bert_dir() -> Optional[Path]:
        """Highest validation PR-AUC among trained text critics.

        Several training runs leave several directories. Picking the first one
        alphabetically is how a 3-second smoke-test model ends up deployed.
        """
        import json

        best, best_score = None, -1.0
        for d in sorted(MODELS.glob("critic_bert_*")):
            meta = d / "meta.json"
            if not meta.exists():
                continue
            try:
                score = json.loads(meta.read_text()).get("val_pr_auc", -1)
            except Exception:                                   # noqa: BLE001
                continue
            if score > best_score:
                best, best_score = d, score
        return best

    @property
    def version(self) -> str:
        parts = []
        if self.lgbm is not None:
            parts.append(f"lgbm{self.w_lgbm:g}")
        if self.bert is not None:
            parts.append(f"bert{self.w_bert:g}")
        return "+".join(parts) or "none"

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def score(self, question: str, steps: Sequence[TrajectoryStep]) -> Optional[float]:
        """P(this run ends wrong), judged at the most recent step.

        None means "no opinion" — the caller must treat that as don't-stop.
        """
        if not steps:
            return None
        try:
            return self._score(question, steps[-1])
        except Exception as e:                                  # noqa: BLE001
            # A broken critic must not take down a run that was going fine.
            print(f"  critic: scoring failed ({type(e).__name__}: {e}) — continuing")
            return None

    def _score(self, question: str, step: TrajectoryStep) -> float:
        row = _step_to_row(question, step)
        df = pd.DataFrame([row])

        p_lgbm = p_bert = None

        if self.lgbm is not None:
            X = engineer(df)
            p_lgbm = float(self.lgbm.predict_proba(X)[0, 1])

        if self.bert is not None:
            p_bert = self._score_text(build_text(row, self.text_format))

        if p_lgbm is None:
            return p_bert
        if p_bert is None:
            return p_lgbm
        return self.w_lgbm * p_lgbm + self.w_bert * p_bert

    def _score_text(self, text: str) -> float:
        import torch

        with torch.no_grad():
            enc = self.tokenizer(text, truncation=True, max_length=self.max_len,
                                 return_tensors="pt").to(self.device)
            logits = self.bert(**enc).logits
            return float(torch.softmax(logits, dim=1)[0, 1].cpu())

    def score_frame(self, df: pd.DataFrame) -> pd.Series:
        """Batch scoring for offline replay — same maths, no Python loop.

        Used by scripts/measure_early_stop.py, which scores 5,461 steps and
        would take minutes one at a time.
        """
        p = pd.Series(0.0, index=df.index)

        if self.lgbm is not None:
            p += self.w_lgbm * self.lgbm.predict_proba(engineer(df))[:, 1]

        if self.bert is not None:
            import numpy as np
            import torch

            texts = [build_text(r, self.text_format) for _, r in df.iterrows()]
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), 64):
                    enc = self.tokenizer(texts[i:i + 64], truncation=True,
                                         max_length=self.max_len, padding=True,
                                         return_tensors="pt").to(self.device)
                    logits = self.bert(**enc).logits
                    out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            p += self.w_bert * np.concatenate(out)

        return p


# ======================================================================
# compatibility layer for orchestration/graph.py (layer D)
# ======================================================================
#
# graph.py imports `Critic` and calls score_step() / should_stop(). That was
# a hand-written rule stub; this keeps the same interface and puts the trained
# model behind it. Do not rename or remove without telling layer D — an import
# error here takes the whole orchestration layer down, which is how this class
# came to exist.
#
# NOTE THE INVERSION. score_step() returns STEP QUALITY, where 1.0 is good —
# that is what graph.py's best-of-N selection assumes when it picks the higher
# score. CriticScorer.score() returns P(FAIL), where 1.0 is bad. Getting this
# backwards would silently make best-of-N choose the worse branch every time,
# with nothing failing anywhere.

_SHARED: Optional["CriticScorer"] = None
_LOAD_FAILED = False


def _shared_scorer() -> Optional[CriticScorer]:
    """One scorer for the whole process.

    graph.py constructs Critic() inside every node call. Loading a 600 MB
    transformer on each of those would be ruinous, so the underlying scorer is
    built once and reused.
    """
    global _SHARED, _LOAD_FAILED
    if _SHARED is None and not _LOAD_FAILED:
        try:
            _SHARED = CriticScorer.load()
        except Exception:                                       # noqa: BLE001
            _LOAD_FAILED = True                                 # don't retry every call
    return _SHARED


class Critic:
    """Layer B's critic, as layer D expects it.

    Falls back to the original heuristics when no trained model is on disk —
    a fresh clone has no models/critic_*.joblib until someone trains one, and
    the orchestration tests must still run in CI.
    """

    def __init__(self, version: str = ""):
        self._scorer = _shared_scorer()
        self.version = version or (self._scorer.version if self._scorer
                                   else "heuristic-fallback")

    def score_step(self, history: Sequence[TrajectoryStep],
                   step: TrajectoryStep) -> float:
        """Quality of this step, 0.0 (harmful) to 1.0 (good)."""
        if self._scorer is not None:
            question = ""       # not available at this call site; the models
                                # handle an empty question, it just weakens them
            p_fail = self._scorer.score(question, list(history) + [step])
            if p_fail is not None:
                return 1.0 - p_fail
        return self._heuristic_score(step)

    @staticmethod
    def _heuristic_score(step: TrajectoryStep) -> float:
        """The original rules, kept as a fallback rather than deleted."""
        if step.observation:
            if step.observation.status == "ok":
                return 0.95 if step.observation.data else 0.80
            return 0.50 if step.observation.hint else 0.20   # recoverable vs not
        if step.action and step.action.is_final:
            return 1.00
        return 0.70

    def should_stop(self, history: Sequence[TrajectoryStep]) -> bool:
        """True when this run looks doomed enough to abandon."""
        if not history:
            return False

        # Repeating an action is pointless whether it worked before or not.
        if history[-1].repeat_count >= 2:
            return True

        if self._scorer is not None:
            p_fail = self._scorer.score("", list(history))
            if p_fail is not None:
                return p_fail > DEFAULT_THRESHOLD

        # Fallback: three consecutive poor steps.
        if len(history) >= 3:
            recent = [self._heuristic_score(s) for s in history[-3:]]
            return all(s < 0.3 for s in recent)
        return False
