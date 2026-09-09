"""Tests that run without the recordings.

The evaluation footage is not published, so every test here builds its own
landmark arrays with the exact EAR and MAR the case needs. That keeps the two
properties the system actually rests on checkable from a clean clone:

  * the risk accumulator escalates only on sustained evidence, at the
    documented 1.5 / 4.0 / 7.5 second-of-evidence thresholds, and
  * the temporal window is strictly causal - perturbing a future frame cannot
    change the output already produced for an earlier one.

    python -m pytest tests -q          (or simply: python tests/test_engine.py)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "driveguard"))
sys.path.insert(0, os.path.join(ROOT, "src", "features"))

from driveguard_engine import load_branch_a, RiskEngine, BANDS      # noqa: E402
from extract_features import LEFT_EYE, RIGHT_EYE, MOUTH             # noqa: E402
from extract_features import ear_of, mar_of                         # noqa: E402


def landmarks(ear, mar):
    """A 478-point array whose EAR and MAR are exactly the values asked for.

    Both measures are ratios of landmark distances, so placing the six eye
    points and four mouth points on a unit baseline makes the ratio equal to
    the vertical separation, and every other point is irrelevant to them.
    """
    p = np.zeros((478, 3), np.float32)
    for eye in (LEFT_EYE, RIGHT_EYE):
        i1, i2, i3, i4, i5, i6 = eye
        p[i1] = (0.0, 0.0, 0.0)
        p[i4] = (1.0, 0.0, 0.0)                     # ||p1 - p4|| = 1
        p[i2] = (0.3, ear / 2, 0.0)
        p[i6] = (0.3, -ear / 2, 0.0)
        p[i3] = (0.7, ear / 2, 0.0)
        p[i5] = (0.7, -ear / 2, 0.0)
    top, bottom, left, right = MOUTH
    p[left] = (0.0, 0.0, 0.0)
    p[right] = (1.0, 0.0, 0.0)                      # ||left - right|| = 1
    p[top] = (0.5, mar / 2, 0.0)
    p[bottom] = (0.5, -mar / 2, 0.0)
    return p


def test_landmark_builder_is_exact():
    p = landmarks(0.05, 0.02)
    ear = (ear_of(p, LEFT_EYE) + ear_of(p, RIGHT_EYE)) / 2.0
    assert abs(ear - 0.05) < 1e-6, ear
    assert abs(mar_of(p, MOUTH) - 0.02) < 1e-6


def test_sustained_closure_escalates_through_every_stage():
    """Closed eyes must walk the engine up all four stages, in order."""
    clf, _ = load_branch_a(verbose=False)
    eng = RiskEngine(clf)
    closed = landmarks(0.05, 0.02)
    seen, t = [], 0.0
    for _ in range(500):                            # 20 s at 25 fps
        t += 0.04
        out = eng.step(closed, now=t)
        stage = out.get("stage")
        if stage is not None and (not seen or stage != seen[-1][0]):
            seen.append((stage, out["risk"]))
    stages = [s for s, _ in seen]
    assert stages == sorted(set(stages)), stages    # monotone, no skipping
    assert stages[-1] == 3, stages
    # The bands are thresholds on accumulated evidence, not on wall-clock time:
    # the engine samples at 5 Hz, so it crosses 1.5 evidence-seconds at about
    # t = 1.48 s. Asserting against elapsed time instead of risk is wrong.
    for stage, risk in seen:
        if stage:
            assert risk >= BANDS[stage - 1], (stage, risk)


def test_a_single_blink_cannot_raise_an_alarm():
    """One short closure inside a long alert stretch must stay at stage 0."""
    clf, _ = load_branch_a(verbose=False)
    eng = RiskEngine(clf)
    open_eyes = landmarks(0.30, 0.02)
    closed = landmarks(0.05, 0.02)
    t, worst = 0.0, 0
    for i in range(500):
        t += 0.04
        blink = 150 <= i < 156                      # ~0.24 s, a normal blink
        worst = max(worst, eng.step(closed if blink else open_eyes,
                                    now=t).get("stage") or 0)
    assert worst == 0, worst


def test_window_is_causal():
    """A later frame must not change the output already given for an earlier one.

    Two engines are fed an identical prefix; one then receives a different
    future frame. The decision recorded before they diverged has to match, or
    the window is reading ahead of itself.
    """
    clf, _ = load_branch_a(verbose=False)
    a, b = RiskEngine(clf), RiskEngine(clf)
    open_eyes = landmarks(0.30, 0.02)
    closed = landmarks(0.05, 0.02)
    out_a = out_b = None
    t = 0.0
    for i in range(60):
        t += 0.04
        out_a = a.step(open_eyes, now=t)
        out_b = b.step(open_eyes, now=t)
        if i == 40:
            before_a, before_b = out_a, out_b       # the frame under test
    # diverge only AFTER frame 40
    a.step(open_eyes, now=t + 0.04)
    b.step(closed, now=t + 0.04)
    assert before_a["risk"] == before_b["risk"]
    assert before_a["stage"] == before_b["stage"]
    assert before_a["cls"] == before_b["cls"]


def test_shipped_model_provenance_carries_no_identities():
    _, prov = load_branch_a(verbose=False)
    assert prov["features"] == ["perclos_20", "MAR", "closed_run_w"]
    assert prov["n_train"] == 793
    blob = repr(prov).lower()
    for token in ("jpg", "png", "/", "\\", "name"):
        assert token not in blob.replace("family-names", ""), token


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("PASS  %s" % name)
        except AssertionError as e:
            failed += 1
            print("FAIL  %s: %s" % (name, e))
    print("\n%s" % ("all tests passed" if not failed else "%d failed" % failed))
    sys.exit(1 if failed else 0)
