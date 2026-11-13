"""One-pass prompt-injection scanners: probability that a memory record is an injection.

Notebook 01 uses the scanner only to choose the score-evading phrasing (attack A5); notebook 02 uses it as a
baseline and as a score feature. The first model in config.SCANNERS that loads is used (Prompt Guard 2 needs the
Llama 4 licence accepted on Hugging Face; the ProtectAI model is ungated) and its name is written to the reports.
"""
from __future__ import annotations

import re

import numpy as np

import config
from serving import DRY_RUN


class HFScanner:
    def __init__(self, model_id: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name, self.device, self.torch = model_id, device, torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
        labels = {i: str(l).lower() for i, l in self.model.config.id2label.items()}
        hit = [i for i, l in labels.items() if "inject" in l or "malicious" in l or "jailbreak" in l]
        self.pos = hit[0] if hit else max(labels)

    def score(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=512,
                               return_tensors="pt").to(self.device)
                p = self.torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, self.pos]
                out.append(p.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


class KeywordScanner:
    """Dry-run stand-in: share of instruction-like keywords. Not a result source."""
    name = "keyword-dry-run"
    WORDS = ("always", "every", "whenever", "rule", "instruction", "must", "bcc", "all ", "any ")

    def score(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.array([min(1.0, sum(w in t.lower() for w in self.WORDS) / 3) + 1e-3 * (len(t) % 7) for t in texts])


def load_scanner(device: str = "cuda"):
    if DRY_RUN:
        return KeywordScanner()
    errors = []
    for m in config.SCANNERS:
        try:
            return HFScanner(m, device)
        except Exception as e:          # gated model without licence access, or a download failure
            errors.append(f"{m}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError("no scanner could be loaded:\n" + "\n".join(errors))
