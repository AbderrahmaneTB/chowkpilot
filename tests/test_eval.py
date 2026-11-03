"""Evaluation helpers: checkpoint scoring, confidence intervals, replay files and the scenario catalogue.

    python -m pytest -q
"""

import os
import tempfile

import numpy as np
import pytest
from scipy import stats

from chowkpilot.eval.benchmark import ci95
from chowkpilot.eval.replay import load_replay, save_replay
from chowkpilot.eval.runner import network
from chowkpilot.rl.select import score
from chowkpilot.scenarios import registry


def test_selection_score_weights_layouts_equally():
    base = {"a@1": 100.0, "a@2": 100.0, "b@1": 60.0}
    delays = {"a@1": 50.0, "a@2": 100.0, "b@1": 30.0}
    # layout a: mean(0.5, 1.0) = 0.75, layout b: 0.5, then equal weight per layout
    assert score(delays, base) == pytest.approx(0.625)
    assert score({"a@1": 50.0}, base) is None  # incomplete validation round is never selected


def test_ci95_matches_scipy():
    x = np.array([12.0, 15.5, 9.8, 14.1, 11.3])
    m, h = ci95(list(x) + [float("nan"), None])  # failed runs are dropped, not counted as zero
    lo, hi = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
    assert m == pytest.approx(x.mean())
    assert (m - h, m + h) == pytest.approx((lo, hi))


def test_replay_roundtrip():
    res = {"frames": [{"t": 0, "v": [1, 104, 205, 90, 11], "s": ["GrGr"]}], "mean_delay": 12.5}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "replays", "r.json.gz")
        save_replay(res, path)
        assert load_replay(path) == res


def test_every_scenario_is_signalised():
    scenarios = registry.load_all()
    assert {"train", "test"} <= {s.split for s in scenarios}
    for s in scenarios:
        assert os.path.isfile(s.net_path), s.id
        nm = network(s)
        assert nm.junctions, f"{s.id} has no controllable signal"
        for jm in nm.junctions:
            assert jm.mask.any(axis=1).all(), f"{s.id}: a phase releases no movement"


def test_readme_claims_follow_the_sign():
    from chowkpilot.eval.readme_results import rel
    assert rel(-12.3, "fixed-time") == "**12% lower** than fixed-time"
    assert rel(4.0, "actuated") == "4% higher than actuated"  # a loss is never worded as a gain


def test_paired_stats():
    from chowkpilot.eval.benchmark import paired_stats
    ours, base = [90.0, 80.0, 70.0, 95.0], [100.0, 100.0, 100.0, 100.0]
    r = paired_stats(ours, base)
    assert (r["wins"], r["n"]) == (4, 4)
    assert r["delay_change_pct"] == pytest.approx(-16.25)  # mean of -10, -20, -30, -5 per cent
    assert r["p_value"] == pytest.approx(stats.ttest_rel(ours, base).pvalue)
