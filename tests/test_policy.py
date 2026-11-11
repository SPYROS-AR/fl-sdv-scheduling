"""Unit tests for src/policy.py (no PyTorch needed)."""

import itertools
import math

import numpy as np
import pytest

import policy
from policy import CONTINUE, SYNC, WAIT, PolicyParams

PAPER = PolicyParams()  # weights and thresholds of the submitted runs


# --------------------------------------------------------------------------- #
# Reference: the decision logic of client.py at commit b6b9d63, transcribed
# line by line, used to check that the refactor did not change any decision.
# --------------------------------------------------------------------------- #
def original_admission(u_crit, bandwidth, prior_gain, rho=10, min_epoch_slack=2.0,
                       bw_threshold=20.0, gain_threshold=0.03):
    max_c_i = max(0.0, 1.0 - u_crit)
    if max_c_i < (min_epoch_slack / rho):
        return "WAIT"
    is_bw_poor = bandwidth < bw_threshold
    if is_bw_poor and prior_gain is not None and prior_gain <= gain_threshold:
        return "WAIT"
    return None


def original_stop(expected_gain, u_crit, drift, rho=10, delta=27.0, beta=1.0, gamma=0.006,
                  tau_continue=0.1, gain_threshold=0.03, drift_threshold=20.0):
    c_comp_marginal = (1.0 / rho) * u_crit
    marginal_utility = delta * expected_gain - beta * c_comp_marginal - gamma * drift
    return (drift > drift_threshold or expected_gain < gain_threshold
            or marginal_utility < tau_continue)


def original_post(bandwidth, final_drift, actual_gain, was_in_continue, continue_count,
                  bw_threshold=20.0, drift_threshold=20.0, gain_threshold=0.03,
                  max_continue_rounds=5):
    is_sync = (bandwidth >= bw_threshold or final_drift > (0.85 * drift_threshold)
               or (was_in_continue == 1 and actual_gain < gain_threshold))
    if is_sync:
        return "SYNC", 0
    continue_count += 1
    if continue_count >= max_continue_rounds:
        return "SYNC", 0
    return "CONTINUE", continue_count


def original_budget(u_crit, rho=10):
    max_c_i = max(0.0, 1.0 - u_crit)
    return min(rho, max(1, int(np.floor(max_c_i * rho))))


# --------------------------------------------------------------------------- #
# Equivalence with the submitted code
# --------------------------------------------------------------------------- #
def test_admission_matches_submitted_code():
    rng = np.random.default_rng(0)
    for _ in range(20000):
        u = float(rng.uniform(0, 1))
        b = float(rng.choice([rng.uniform(0, 40), 20.0]))
        g = None if rng.random() < 0.2 else float(rng.choice([rng.uniform(0, 0.1), 0.03]))
        expected = original_admission(u, b, g)
        got = policy.admission(u, b, g, PAPER)
        assert (got is None) == (expected is None), (u, b, g)


def test_legacy_stop_rule_matches_submitted_code():
    rng = np.random.default_rng(1)
    for _ in range(20000):
        g = float(rng.uniform(0, 0.2))
        u = float(rng.uniform(0, 0.95))
        d = float(rng.uniform(0, 25))
        assert policy.should_stop(g, u, d, PAPER) == original_stop(g, u, d)


def test_post_training_decision_matches_submitted_code():
    for b, d, g, cont, k in itertools.product(
        [5.0, 19.99, 20.0, 50.0], [1.0, 16.9, 17.1, 25.0], [0.0, 0.029, 0.03, 0.5],
        [False, True], [0, 1, 3, 4],
    ):
        status, k_new, reason = policy.post_training_decision(b, d, g, cont, k, PAPER)
        exp_status, exp_k = original_post(b, d, g, int(cont), k)
        assert (status, k_new) == (exp_status, exp_k)
        assert (reason == "") == (status == CONTINUE)


def test_slack_budget_matches_submitted_code_where_training_is_admitted():
    for u in np.linspace(0.0, 0.8, 801):
        if policy.admission(float(u), 50.0, None, PAPER) is None:
            assert max(1, policy.slack_epoch_budget(float(u), 10)) == original_budget(float(u))


def test_ema_matches_submitted_formula():
    history = [2.0]
    for loss in [1.5, 1.2, 1.1]:
        expected = 0.3 * loss + 0.7 * history[-1]
        assert policy.ema_append(history, loss, 0.3) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Properties discussed in the audit
# --------------------------------------------------------------------------- #
def test_legacy_utility_test_is_redundant():
    """With the paper's weights, Delta U < tau never decides the stop on its own.

    If G_hat >= G_th and D <= D_max then Delta U >= 27*0.03 - 1.0*0.095 - 0.006*20
    = 0.595 > 0.1, for every U_crit <= 0.95 (the cap of the U_crit formula).
    """
    lower_bound = PAPER.delta * PAPER.gain_threshold - PAPER.beta * 0.95 / PAPER.rho \
        - PAPER.gamma * PAPER.drift_threshold
    assert lower_bound == pytest.approx(0.595)
    assert lower_bound > PAPER.tau_continue
    rng = np.random.default_rng(2)
    for _ in range(20000):
        g = float(rng.uniform(PAPER.gain_threshold, 1.0))
        u = float(rng.uniform(0, 0.95))
        d = float(rng.uniform(0, PAPER.drift_threshold))
        assert policy.marginal_utility(g, u, d, PAPER) >= PAPER.tau_continue


def test_gain_floor_rises_with_load_and_drift():
    p = PolicyParams(stop_rule="utility", tau_continue=0.75)
    assert policy.gain_floor(0.6, 2.0, p) > policy.gain_floor(0.3, 2.0, p)
    assert policy.gain_floor(0.4, 10.0, p) > policy.gain_floor(0.4, 1.0, p)
    # tau = 0.75 places the floor near the legacy G_th = 0.03 at typical load and drift
    assert policy.gain_floor(0.4, 2.0, p) == pytest.approx((0.75 + 0.04 + 0.012) / 27)
    assert 0.028 < policy.gain_floor(0.4, 2.0, p) < 0.032


def test_utility_rule_uses_the_floor_only():
    p = PolicyParams(stop_rule="utility", tau_continue=0.75)
    floor = policy.gain_floor(0.5, 3.0, p)
    assert policy.should_stop(floor * 0.99, 0.5, 3.0, p)
    assert not policy.should_stop(floor * 1.01, 0.5, 3.0, p)
    assert policy.should_stop(1.0, 0.5, p.drift_threshold + 0.1, p)  # drift cap still applies


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        PolicyParams(stop_rule="greedy")
    with pytest.raises(ValueError):
        PolicyParams(delta=0.0)
    with pytest.raises(ValueError):
        PolicyParams(gain_estimate="median")


# --------------------------------------------------------------------------- #
# Admission and transmission rules
# --------------------------------------------------------------------------- #
def test_admission_reasons():
    assert policy.admission(0.85, 50.0, None, PAPER) == policy.WAIT_COMPUTE  # slack < 2/rho
    assert policy.admission(0.79, 50.0, None, PAPER) is None
    assert policy.admission(0.4, 5.0, None, PAPER) is None  # first round: no prior gain
    assert policy.admission(0.4, 5.0, 0.02, PAPER) == policy.WAIT_LINK
    assert policy.admission(0.4, 5.0, 0.05, PAPER) is None  # weak link but still learning
    assert policy.admission(0.4, 5.0, 0.05, PAPER, no_continue=True) == policy.WAIT_LINK


def test_sync_reasons_and_staleness():
    assert policy.post_training_decision(25.0, 1.0, 0.1, False, 0, PAPER)[2] == policy.SYNC_LINK
    assert policy.post_training_decision(5.0, 18.0, 0.1, False, 0, PAPER)[2] == policy.SYNC_DRIFT
    assert policy.post_training_decision(5.0, 1.0, 0.01, True, 1, PAPER)[2] == policy.SYNC_GAIN
    assert policy.post_training_decision(5.0, 1.0, 0.01, False, 0, PAPER)[:2] == (CONTINUE, 1)
    # K_max = 5: at most four consecutive CONTINUE rounds
    k, statuses = 0, []
    for _ in range(6):
        status, k, _ = policy.post_training_decision(5.0, 1.0, 0.2, k > 0, k, PAPER)
        statuses.append(status)
    assert statuses[:5] == [CONTINUE] * 4 + [SYNC]


# --------------------------------------------------------------------------- #
# Curve fit
# --------------------------------------------------------------------------- #
def test_prediction_is_bounded_and_decreasing_on_a_decaying_curve():
    raw = [2.0 * math.exp(-0.5 * e) + 0.5 for e in range(6)]
    history = [raw[0]]
    for loss in raw[1:]:
        policy.ema_append(history, loss, 0.3)
    pred = policy.predict_next_loss(history)
    assert 0.0 <= pred <= history[-1]
    assert history[-1] - pred > 0.0


def original_predict_loss(history):
    """client._predict_loss at commit b6b9d63, verbatim (as a free function)."""
    import warnings

    import scipy.optimize
    from scipy.optimize import OptimizeWarning

    def exp_curve(x, l_inf, a, b):
        return l_inf + a * np.exp(-b * x)

    if len(history) < 3:
        return float(history[-1])
    x_data = np.arange(len(history), dtype=np.float64)
    y_data = np.array(history, dtype=np.float64)
    l_inf_init = max(0.0, float(np.min(y_data) * 0.8))
    a_init = max(1e-3, float(y_data[0] - l_inf_init))
    p0 = [l_inf_init, a_init, 0.5]
    bounds = ([0.0, 0.0, 1e-4], [float(np.min(y_data)), np.inf, 10.0])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = scipy.optimize.curve_fit(exp_curve, x_data, y_data, p0=p0,
                                               bounds=bounds, maxfev=1500)
        pred = float(exp_curve(float(len(history)), *popt))
        return min(float(y_data[-1]), max(0.0, pred))
    except (RuntimeError, ValueError, FloatingPointError):
        delta_last = float(y_data[-1] - y_data[-2])
        if delta_last < 0:
            return max(0.0, float(y_data[-1] + 0.5 * delta_last))
        return float(y_data[-1])


def test_predict_next_loss_matches_submitted_code():
    rng = np.random.default_rng(3)
    for _ in range(300):
        n = int(rng.integers(3, 9))
        raw = 0.3 + rng.uniform(0.5, 2.0) * np.exp(-rng.uniform(0.05, 1.5) * np.arange(n))
        raw = raw + rng.normal(0, rng.choice([0.0, 0.02, 0.1]), n)
        history = [float(raw[0])]
        for loss in raw[1:]:
            policy.ema_append(history, float(loss), 0.3)
        assert policy.predict_next_loss(history) == original_predict_loss(history)


def test_legacy_gain_estimate_is_biased_on_a_rising_loss():
    """Audit finding: the fitted curve can only decrease, so on a rising loss it regresses
    to the mean and H[e] - L_hat(e+1) reports a large spurious gain."""
    rising = [1.0, 1.1, 1.2]
    legacy = policy.expected_gain(rising, PAPER)
    assert legacy == pytest.approx(0.10, abs=0.005)
    assert legacy > PAPER.gain_threshold  # the legacy rule keeps training
    assert not policy.should_stop(legacy, 0.4, 2.0, PAPER)


def test_curve_gain_estimate_is_zero_on_flat_and_rising_losses():
    curve = PolicyParams(gain_estimate="curve")
    assert policy.expected_gain([1.0, 1.0, 1.0], curve) < 1e-3
    assert policy.expected_gain([1.0, 1.1, 1.2], curve) < 1e-3
    assert policy.expected_gain([1.0, 1.03, 1.081], curve) < 1e-3  # EMA of 1.0, 1.1, 1.2


def test_gain_estimates_agree_on_a_decaying_loss():
    raw = [1.5 * math.exp(-0.6 * e) + 0.4 for e in range(4)]
    history = [raw[0]]
    for loss in raw[1:]:
        policy.ema_append(history, loss, 0.3)
    legacy = policy.expected_gain(history, PAPER)
    curve = policy.expected_gain(history, PolicyParams(gain_estimate="curve"))
    assert legacy > PAPER.gain_threshold and curve > PAPER.gain_threshold
    assert 0.67 < curve / legacy < 1.5


def test_expected_gain_needs_three_points():
    assert policy.expected_gain([1.3, 1.2], PAPER) == 0.0


def test_short_history_returns_last_value():
    assert policy.predict_next_loss([1.3, 1.2]) == 1.2


# --------------------------------------------------------------------------- #
# Baselines and seeding
# --------------------------------------------------------------------------- #
def test_baseline_epochs():
    assert policy.baseline_epochs("fixed", 10, 0.7, 10) == 10  # ignores the slack
    assert policy.baseline_epochs("fixedcap", 10, 0.7, 10) == 3
    assert policy.baseline_epochs("fixedcap", 2, 0.3, 10) == 2
    assert policy.baseline_epochs("slackfill", None, 0.3, 10) == 7
    assert policy.baseline_epochs("threshold", 5, 0.6, 10) == 4
    assert policy.baseline_epochs("randpart", 5, 0.9, 10) == 5
    with pytest.raises(ValueError):
        policy.baseline_epochs("greedy", 5, 0.5, 10)


def test_participation_is_deterministic_and_has_the_right_rate():
    draws = [policy.participates(3, r, v, 0.72) for r in range(1, 401) for v in range(10)]
    again = [policy.participates(3, r, v, 0.72) for r in range(1, 401) for v in range(10)]
    assert draws == again
    assert abs(np.mean(draws) - 0.72) < 0.03
    other_seed = [policy.participates(4, r, v, 0.72) for r in range(1, 401) for v in range(10)]
    assert draws != other_seed


def test_derive_seed_is_stable():
    assert policy.derive_seed(1, 2, 3) == policy.derive_seed(1, 2, 3)
    assert policy.derive_seed(1, 2, 3) != policy.derive_seed(1, 2, 4)
