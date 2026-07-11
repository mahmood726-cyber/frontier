"""Regression tests for the frontier statistical core (src/).

These exercise the pure, dependency-light functions (path resolution, the
closed-form Wasserstein-over-Wasserstein distance, and the conformal
meta-learning input guards). They intentionally do NOT call
run_svgd_inference, which requires a working C++/pytensor toolchain.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import model_frontier as mf  # noqa: E402
import ingest_data as ingest  # noqa: E402


# --- F1: entry-point paths resolve to the repo root (no 'frontier/' prefix) ---

def test_paths_resolve_to_repo_root():
    assert mf.ROOT == PROJECT_ROOT
    assert ingest.ROOT == PROJECT_ROOT
    # The real input file lives at data/, not a nested frontier/data/.
    assert (mf.ROOT / "data" / "frontier_synthesis_input.json").exists()
    assert not (PROJECT_ROOT / "frontier").exists()


# --- F2: WoW distance is deterministic, noise-free, and correctly ordered ---

def test_wow_distance_is_deterministic():
    p1 = {"age_mean": 62, "age_std": 10}
    p2 = {"age_mean": 64, "age_std": 12}
    a = mf.calculate_wow_distance(p1, p2)
    b = mf.calculate_wow_distance(p1, p2)
    assert a == b  # exact reproducibility, no Monte Carlo jitter


def test_wow_distance_matches_scipy_wasserstein():
    from scipy.stats import wasserstein_distance

    p1 = {"age_mean": 62, "age_std": 10}
    p2 = {"age_mean": 60, "age_std": 11}
    rng = np.random.default_rng(0)
    s1 = rng.normal(p1["age_mean"], p1["age_std"], 2_000_000)
    s2 = rng.normal(p2["age_mean"], p2["age_std"], 2_000_000)
    mc = wasserstein_distance(s1, s2)
    # Empirical W1 has a slow-converging positive bias; a loose bound still
    # catches a wrong metric (e.g. W2 here would be ~2.236, off by ~0.22).
    assert mf.calculate_wow_distance(p1, p2) == pytest.approx(mc, abs=3.5e-2)


def test_wow_zero_scale_difference_is_mean_gap():
    # sigma == 0 branch: identical spreads -> W1 collapses to |mean gap|.
    p1 = {"age_mean": 62, "age_std": 10}
    p2 = {"age_mean": 65, "age_std": 10}
    assert mf.calculate_wow_distance(p1, p2) == pytest.approx(3.0)


def test_wow_identifies_china_as_most_similar():
    # The micro-paper's S4 conclusion must be reproducible, not a coin flip.
    base = {"age_mean": 62, "age_std": 10}
    obs = [
        {"location": "USA", "age_mean": 64, "age_std": 12},
        {"location": "IND", "age_mean": 55, "age_std": 15},
        {"location": "CHN", "age_mean": 60, "age_std": 11},
        {"location": "NGA", "age_mean": 52, "age_std": 14},
    ]
    dists = {o["location"]: mf.calculate_wow_distance(base, o) for o in obs}
    assert min(dists, key=dists.get) == "CHN"


# --- F6: conformal meta-learning guards degenerate inputs instead of NaN/inf ---

_ANCHORS = [{"hr": 0.81}, {"hr": 0.83}]


def test_conformal_empty_anchors_raises():
    with pytest.raises(ValueError):
        mf.run_conformal_meta_learning([], [{"location": "X", "hr_obs": 0.8, "label_quality": 0.7}])


def test_conformal_zero_label_quality_raises():
    obs = [{"location": "X", "hr_obs": 0.8, "label_quality": 0.0}]
    with pytest.raises(ValueError):
        mf.run_conformal_meta_learning(_ANCHORS, obs)


def test_conformal_missing_label_quality_raises():
    obs = [{"location": "X", "hr_obs": 0.8}]
    with pytest.raises(ValueError):
        mf.run_conformal_meta_learning(_ANCHORS, obs)


def test_conformal_single_anchor_gives_defined_output():
    obs = [{"location": "X", "hr_obs": 0.8, "label_quality": 0.7}]
    out = mf.run_conformal_meta_learning([{"hr": 0.81}], obs)
    assert len(out) == 1
    row = out[0]
    for key in ("conformal_low", "conformal_high", "estimate"):
        assert np.isfinite(row[key])
    # k=1 -> zero-width nonconformity band, a defined point interval (not inf/nan).
    assert row["conformal_low"] == row["conformal_high"]


def test_conformal_valid_input_bounds_unchanged():
    # Guards must not perturb numeric output for the shipped, valid dataset.
    obs = [{"location": "USA", "hr_obs": 0.85, "label_quality": 0.7}]
    out = mf.run_conformal_meta_learning(_ANCHORS, obs)
    assert out[0]["conformal_low"] == pytest.approx(0.8357, abs=1e-4)
    assert out[0]["conformal_high"] == pytest.approx(0.8643, abs=1e-4)
