"""Attack A4 (upgrade-targeted) and A5 (score-evading) selection (plan v1, Section 7.3).

For every A4 record the eight phrasings were probed on the old reader of the record's group (J_VARIANT_OLD probes)
and on the group's surrogate of the new reader (J_VARIANT_SURR probes). A phrasing is eligible when it is retrieved
by at least one query of its category, the old reader never acts on it and the surrogate does. A4 keeps the first
eligible phrasing; A5 keeps the eligible phrasing with the lowest scanner score. Records with no eligible phrasing
keep phrasing 0 (A4) or the lowest-scoring phrasing (A5) and are flagged, so they can be reported apart.
The real new reader is never consulted: A4 and A5 are transfer attacks.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

import config


def variant_outcomes(episodes: list[dict]) -> dict[tuple[str, int], dict]:
    """(record, phrasing) -> {"hits": number of probes with the attacker outcome, "n": probes} for one reader."""
    out: dict[tuple[str, int], dict] = defaultdict(lambda: {"hits": 0, "n": 0})
    for e in episodes:
        if e["kind"] != "variant":
            continue
        d = out[(e["probe_record"], e["variant"])]
        d["n"] += 1
        d["hits"] += int(bool(e["attack_hit"]))
    return dict(out)


def select(suite: dict, outcomes_by_reader: dict[str, dict], scanner_score: dict[str, float],
           retrieved: dict[tuple[str, int], bool]) -> pd.DataFrame:
    rows = []
    for rid, r in sorted(suite["records"].items()):
        if not (r["poison"] and r["attack"] == "A4"):
            continue
        g = config.A4_GROUPS[r["group"]]
        old, sur = outcomes_by_reader[g["old"]], outcomes_by_reader[g["surrogate"]]
        info = []
        for v, text in enumerate(r["variants"]):
            ret = bool(retrieved[(rid, v)])
            o, s = old.get((rid, v)), sur.get((rid, v))
            if ret and (o is None or s is None):
                raise KeyError(f"phrasing {v} of {rid} has not been probed on {g['old']} and {g['surrogate']}")
            inert = bool(o and o["hits"] == 0)          # phrasings never retrieved are not probed and not eligible
            active = bool(s and s["hits"] > 0)
            info.append({"v": v, "inert_old": inert, "active_surrogate": active, "retrieved": ret,
                         "eligible": inert and active and ret, "score": float(scanner_score[text])})
        elig = [x for x in info if x["eligible"]]
        a4 = elig[0]["v"] if elig else 0
        if elig:
            a5 = min(elig, key=lambda x: (x["score"], x["v"]))["v"]
        else:
            pool = [x for x in info if x["active_surrogate"] and x["retrieved"]] or info
            a5 = min(pool, key=lambda x: (x["score"], x["v"]))["v"]
        rows.append({"record_id": rid, "user_id": r["user_id"], "group": r["group"], "underlying": r["underlying"],
                     "category": r["category"], "n_eligible": len(elig), "a4_variant": a4, "a4_eligible": bool(elig),
                     "a5_variant": a5, "a5_eligible": bool(elig), "a4_score": info[a4]["score"],
                     "a5_score": info[a5]["score"], "n_inert_old": sum(x["inert_old"] for x in info),
                     "n_active_surrogate": sum(x["active_surrogate"] for x in info),
                     "n_retrieved": sum(x["retrieved"] for x in info)})
    return pd.DataFrame(rows)
