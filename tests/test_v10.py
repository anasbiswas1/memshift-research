"""CPU tests of notebook 10: the core extension of amendment 4."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MEMSHIFT_REPO_ROOT", str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

import config  # noqa: E402
import policies as PO  # noqa: E402

EXPOSED = {("r1", "email"), ("r1", "payments"), ("r2", "calendar"), ("r3", "email"), ("r4", "documents")}
K5 = frozenset({("r1", "email"), "r9"})                                   # the loosest candidate: one scored unit, one census positive
K1 = K5 | frozenset({("r2", "calendar"), ("r3", "email"), ("r4", "documents")})   # the strictest: more scored units


def test_core_extension_uses_the_loosest_candidates_records_only():
    ext = PO.core_extension(K1, K5, EXPOSED)
    assert K1 <= ext
    assert ("r1", "calendar") in ext and ("r1", "documents") in ext            # zero-exposure units of a K5 record
    assert ("r1", "payments") not in ext                                         # exposed and kept: untouched
    for r in ("r2", "r3", "r4"):                                                 # K1-only records are not extended
        assert not any(isinstance(u, tuple) and u[0] == r and u not in K1 for u in ext)
    assert "r9" in ext and not any(isinstance(u, tuple) and u[0] == "r9" for u in ext)


def test_core_extension_is_between_the_candidate_and_the_full_extension():
    ext = PO.core_extension(K1, K5, EXPOSED)
    full = PO.extend_zero_exposure(K1, EXPOSED)
    assert K1 <= ext <= full
    assert PO.core_extension(K5, K5, EXPOSED) == PO.extend_zero_exposure(K5, EXPOSED)   # equal at the loosest candidate
    assert PO.core_extension(ext, K5, EXPOSED) == ext                                    # idempotent


def test_core_extension_leaves_exposed_units_alone():
    ext = PO.core_extension(K1, K5, EXPOSED)
    for r, c in EXPOSED:
        assert PO.in_policy(r, c, ext) == PO.in_policy(r, c, K1)


def test_core_extension_rejects_a_non_subset():
    with pytest.raises(AssertionError):
        PO.core_extension(K5, K1, EXPOSED)


def test_preregistration_has_notebook_10():
    p = config.preregistration()["notebook_10"]
    assert p["users"] == [190, 219] and "C10" in p["claims"]
    assert config.CORE_EXTENSION_CLAIMS["C10_max_heldout_success_loss"] == 0.01
