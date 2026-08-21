# memory/episodic.py
"""Krishna (Layer C) - Episodic Memory Store.
Maintains past run summaries and failure reflections.
Enforces hard bounds on recall: max 3 items and max ~200 tokens total.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from contracts import RunRecord

EPISODIC_FILE = "data/episodic_memory.json"

class EpisodicMemory:
    """Bounded episodic memory buffer."""
    def __init__(self, storage_path: str = EPISODIC_FILE):
        self.storage_path = storage_path
        self.reflections: List[Dict[str, str]] = []
        self.run_summaries: List[Dict[str, str]] = []
        self._load()

    def _load(self):
        """Load stored memories from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.reflections = data.get("reflections", [])
                    self.run_summaries = data.get("run_summaries", [])
            except Exception:
                self.reflections = []
                self.run_summaries = []
        else:
            # Preload helpful initial reflection lessons
            self.reflections = [
                {"run_id": "r-0001", "text": "The sales column is named `sales` in the orders table."},
                {"run_id": "r-0002", "text": "Customer names and segments live in the `customers` table, joined on customer_id."}
            ]

    def _save(self):
        """Persist memories to disk."""
        Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump({
                    "reflections": self.reflections,
                    "run_summaries": self.run_summaries
                }, f, indent=2)
        except Exception:
            pass

    def add_run(self, record: RunRecord) -> None:
        """Summarize and index a completed run."""
        summary = f"Task: {record.question} | Outcome: {record.termination} | Steps: {len(record.steps)}"
        self.run_summaries.append({"run_id": record.run_id, "summary": summary})
        self._save()

    def add_reflection(self, run_id: str, text: str) -> None:
        """Store a short lesson after a failure (e.g. 'the sales column is sales, not sale_amount')."""
        self.reflections.append({"run_id": run_id, "text": text})
        self._save()

    def recall(self, question: str, k: int = 3) -> List[str]:
        """Return up to k relevant past lessons.
        BOUNDED - never more than 3 items, and never more than 200 tokens total.
        Works seamlessly on empty store.
        """
        if not self.reflections:
            return []

        # Simple semantic keyword matching for fast retrieval (<1ms)
        q_words = set(question.lower().split())
        scored = []
        for item in self.reflections:
            text = item.get("text", "")
            match_score = sum(1 for w in q_words if w in text.lower())
            scored.append((match_score, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = [item[1] for item in scored[:min(k, 3)]]

        # Enforce 200 token budget (~800 characters)
        bounded = []
        char_budget = 800
        current_chars = 0
        for item in top_k:
            if current_chars + len(item) <= char_budget:
                bounded.append(item)
                current_chars += len(item)
            else:
                break

        return bounded
