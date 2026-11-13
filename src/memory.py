"""Memory stores, embedders, top-k retrieval and exposure (plan v1, Equation 1).

A store maps every record id of the service to the text it holds; store variants differ only in which phrasing an
A4 record carries ("main": the A4 choice, "a5": the score-evading choice) or in leaving poison out ("clean").
Retrieval is per user: a query only sees its own user's records. Embeddings are cached on disk by text hash, so
store variants and reruns reuse them.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import config


def text_key(t: str) -> str:
    return hashlib.sha1(t.encode()).hexdigest()


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for CPU tests and dry runs. `variant` changes the features so
    that two retrievers rank records differently, as real embedding models do. Not a result source."""

    def __init__(self, variant: str, dim: int = 384):
        self.variant, self.dim = variant, dim

    def _feats(self, t: str):
        w = re.findall(r"[a-z0-9]+", t.lower())
        feats = list(w)
        if self.variant.endswith("v15") or self.variant.startswith("bge_base"):
            feats += [a + "_" + b for a, b in zip(w, w[1:])]
        if self.variant.startswith("minilm"):
            feats = [x[:4] for x in w]
        return feats

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f in self._feats(t):
                h = int(hashlib.md5(f"{self.variant}|{f}".encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


class STEmbedder:
    """A sentence-transformers model with the retriever's query instruction (BGE) and normalised embeddings."""

    def __init__(self, tag: str, device: str = "cuda", batch_size: int = 128):
        from sentence_transformers import SentenceTransformer
        self.tag, self.batch_size = tag, batch_size
        self.model = SentenceTransformer(config.RETRIEVERS[tag], device=device)
        self.instruction = config.QUERY_INSTRUCTION[tag]

    def encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        if is_query and self.instruction:
            texts = [self.instruction + t for t in texts]
        v = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)
        return v.astype(np.float32)


class EmbeddingCache:
    """Vectors keyed by (role, text hash), one file per retriever under data/embeddings/."""

    def __init__(self, tag: str, root: Path | None = None):
        self.tag = tag
        self.path = Path(root or (config.DATA / "embeddings")) / f"{tag}.npz"
        self.vec: dict[str, np.ndarray] = {}
        if self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            for k, v in zip(z["keys"], z["vecs"]):
                self.vec[str(k)] = v

    def ensure(self, embedder, texts: list[str], is_query: bool) -> int:
        role = "q" if is_query else "r"
        missing = sorted({t for t in texts if f"{role}:{text_key(t)}" not in self.vec})
        if missing:
            vs = embedder.encode(missing, is_query)
            for t, v in zip(missing, vs):
                self.vec[f"{role}:{text_key(t)}"] = v
        return len(missing)

    def get(self, texts: list[str], is_query: bool) -> np.ndarray:
        role = "q" if is_query else "r"
        return np.stack([self.vec[f"{role}:{text_key(t)}"] for t in texts])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.vec.keys()))
        vecs = np.stack(list(self.vec.values())) if self.vec else np.zeros((0, 1), np.float32)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, vecs=vecs)
        tmp.replace(self.path)


def store_texts(suite: dict, variant_choice: dict | None = None, include_poison: bool = True) -> dict[str, str]:
    """record_id -> text. variant_choice maps A4 record ids to the phrasing index they carry."""
    out = {}
    for rid, r in suite["records"].items():
        if r["poison"] and not include_poison:
            continue
        if r["poison"] and variant_choice and rid in variant_choice:
            out[rid] = r["variants"][variant_choice[rid]]
        else:
            out[rid] = r["text"]
    return out


def all_texts(suite: dict) -> tuple[list[str], list[str]]:
    """Every record text and phrasing variant the service can hold, and every query text."""
    recs = set()
    for r in suite["records"].values():
        recs.add(r["text"])
        if r["poison"] and r["attack"] == "A4":
            recs.update(r["variants"])
    return sorted(recs), sorted({q["text"] for q in suite["queries"].values()})


class Retrieval:
    """Per-user similarity matrices and top-k lists for one store under one retriever."""

    def __init__(self, suite: dict, texts: dict[str, str], cache: EmbeddingCache, k: int = config.TOP_K):
        self.k, self.texts = k, texts
        self.by_user: dict[int, dict] = {}
        recs_by_user, qs_by_user = defaultdict(list), defaultdict(list)
        for rid in texts:
            recs_by_user[suite["records"][rid]["user_id"]].append(rid)
        for qid, q in suite["queries"].items():
            qs_by_user[q["user_id"]].append(qid)
        for uid, rids in recs_by_user.items():
            qids = qs_by_user[uid]
            R = cache.get([texts[r] for r in rids], False)
            Q = cache.get([suite["queries"][q]["text"] for q in qids], True)
            S = Q @ R.T
            order = np.argsort(-S, axis=1, kind="stable")[:, :k]
            self.by_user[uid] = {"rids": rids, "qids": qids, "S": S, "topk": order,
                                 "ridx": {r: i for i, r in enumerate(rids)}, "qidx": {q: i for i, q in enumerate(qids)}}

    def topk(self, query_id: str, uid: int) -> list[str]:
        u = self.by_user[uid]
        return [u["rids"][j] for j in u["topk"][u["qidx"][query_id]]]

    def sim(self, query_id: str, record_id: str, uid: int) -> float:
        u = self.by_user[uid]
        return float(u["S"][u["qidx"][query_id], u["ridx"][record_id]])

    def exposure_counts(self, suite: dict) -> dict[tuple[str, str], int]:
        """(record_id, category) -> number of the service's queries of that category that retrieve the record."""
        cnt: dict[tuple[str, str], int] = defaultdict(int)
        for uid, u in self.by_user.items():
            for qi, qid in enumerate(u["qids"]):
                cat = suite["queries"][qid]["category"]
                for j in u["topk"][qi]:
                    cnt[(u["rids"][j], cat)] += 1
        return dict(cnt)


def topk_with_override(suite: dict, ret: Retrieval, cache: EmbeddingCache, uid: int, query_id: str,
                       record_id: str, new_text: str) -> tuple[list[str], float]:
    """Top-k for one query when one record carries a different phrasing; also the record's similarity."""
    u = ret.by_user[uid]
    qi = u["qidx"][query_id]
    s = u["S"][qi].copy()
    j = u["ridx"][record_id]
    qv = cache.get([suite["queries"][query_id]["text"]], True)[0]
    s[j] = float(cache.get([new_text], False)[0] @ qv)
    order = np.argsort(-s, kind="stable")[:ret.k]
    return [u["rids"][i] for i in order], float(s[j])
