"""Decision logic of the three-state policy engine, free of PyTorch.

Every function here maps to one step of Algorithm 1 in the paper, so the rules can
be unit-tested (tests/test_policy.py) and cited independently of the training loop
in client.py. The baseline rules used by the ablation experiments live here too.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.optimize
from scipy.optimize import OptimizeWarning

WAIT, CONTINUE, SYNC = "WAIT", "CONTINUE", "SYNC"

# Reasons reported in the per-client log (clients_<exp>.csv)
WAIT_COMPUTE, WAIT_LINK, WAIT_NOT_SELECTED = "compute", "link", "not_selected"
SYNC_LINK, SYNC_DRIFT, SYNC_GAIN, SYNC_STALENESS, SYNC_ALWAYS = (
    "link",
    "drift",
    "gain_exhausted",
    "staleness",
    "always",
)

STOP_RULES = ("legacy", "utility")
GAIN_ESTIMATES = ("last", "curve")


@dataclass(frozen=True)
class PolicyParams:
    """Thresholds and weights of the engine (defaults = the values used in the paper)."""

    rho: int = 10
    delta: float = 27.0
    beta: float = 1.0
    gamma: float = 0.006
    tau_continue: float = 0.1
    gain_threshold: float = 0.03
    drift_threshold: float = 20.0
    bw_threshold: float = 20.0
    min_epoch_slack: float = 2.0
    max_continue_rounds: int = 5
    drift_sync_fraction: float = 0.85
    ema_weight: float = 0.3
    stop_rule: str = "legacy"
    gain_estimate: str = "last"

    def __post_init__(self):
        if self.stop_rule not in STOP_RULES:
            raise ValueError(f"stop_rule must be one of {STOP_RULES}")
        if self.gain_estimate not in GAIN_ESTIMATES:
            raise ValueError(f"gain_estimate must be one of {GAIN_ESTIMATES}")
        if self.rho <= 0 or self.delta <= 0:
            raise ValueError("rho and delta must be positive")


# --------------------------------------------------------------------------- #
# Compute slack (constraints (10)-(11))
# --------------------------------------------------------------------------- #
def slack_epoch_budget(u_crit: float, rho: int) -> int:
    """Largest number of epochs that fits in the slack, floor(rho * (1 - U_crit)).

    Kept bit-identical to the original implementation (no epsilon), so the budget
    for a given trace sample is the same as in the submitted runs.
    """
    c_max = max(0.0, 1.0 - float(u_crit))
    return int(min(rho, np.floor(c_max * rho)))


# --------------------------------------------------------------------------- #
# Stage 1: admission (WAIT before any computation)
# --------------------------------------------------------------------------- #
def admission(
    u_crit: float,
    bandwidth: float,
    prior_gain: float | None,
    p: PolicyParams,
    no_continue: bool = False,
) -> str | None:
    """Return a WAIT reason, or None if the vehicle may start training.

    (i)  compute: slack cannot hold the two epochs needed for the first curve fit;
    (ii) link:    weak uplink and the previous training window gained little.
    With ``no_continue`` (ablation) a weak uplink alone forces WAIT, i.e. the
    vehicle only trains in rounds in which it can upload.
    """
    if max(0.0, 1.0 - float(u_crit)) < p.min_epoch_slack / p.rho:
        return WAIT_COMPUTE
    weak_link = bandwidth < p.bw_threshold
    if weak_link and no_continue:
        return WAIT_LINK
    if weak_link and prior_gain is not None and prior_gain <= p.gain_threshold:
        return WAIT_LINK
    return None


# --------------------------------------------------------------------------- #
# Stage 2: per-epoch stopping
# --------------------------------------------------------------------------- #
def ema_append(history: list[float], loss: float, weight: float) -> float:
    """Append the EMA-smoothed loss L_s(e) = w*L(e) + (1-w)*L_s(e-1); return it."""
    smoothed = weight * float(loss) + (1.0 - weight) * float(history[-1])
    history.append(smoothed)
    return smoothed


def _exp_curve(x, l_inf, a, b):
    return l_inf + a * np.exp(-b * x)


def _fit_curve(history: list[float]):
    """Bounded least-squares fit of L(e) = L_inf + a*exp(-b*e); None if the solver fails.

    Bounds: 0 <= L_inf <= min(H), a >= 0, 1e-4 <= b <= 10 (trust-region reflective,
    at most 1500 evaluations), started from p0 = [0.8 min H, max(1e-3, H0 - 0.8 min H), 0.5].
    """
    x_data = np.arange(len(history), dtype=np.float64)
    y_data = np.asarray(history, dtype=np.float64)
    l_inf_init = max(0.0, float(np.min(y_data) * 0.8))
    a_init = max(1e-3, float(y_data[0] - l_inf_init))
    p0 = [l_inf_init, a_init, 0.5]
    bounds = ([0.0, 0.0, 1e-4], [float(np.min(y_data)), np.inf, 10.0])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = scipy.optimize.curve_fit(
                _exp_curve, x_data, y_data, p0=p0, bounds=bounds, maxfev=1500
            )
        return popt
    except (RuntimeError, ValueError, FloatingPointError):
        return None


def predict_next_loss(history: list[float]) -> float:
    """Predicted loss at epoch e+1, clipped to [0, H[-1]] (identical to the submitted code).

    Fallback when the fit fails: extrapolate the last decrease at half rate.
    """
    if len(history) < 3:
        return float(history[-1])
    popt = _fit_curve(history)
    if popt is not None:
        pred = float(_exp_curve(float(len(history)), *popt))
        return min(float(history[-1]), max(0.0, pred))
    delta_last = float(history[-1] - history[-2])
    if delta_last < 0:
        return max(0.0, float(history[-1] + 0.5 * delta_last))
    return float(history[-1])


def expected_gain(history: list[float], p: PolicyParams) -> float:
    """Predicted validation gain of one more epoch, G_hat(e+1).

    last  : H[e] - L_hat(e+1), as in the submitted runs. Because the fitted curve can only
            decrease, on a flat or rising (over-fitting) loss the fit regresses to the mean
            and this difference is spuriously positive (e.g. 0.10 for H = [1.0, 1.1, 1.2]).
    curve : f(e) - f(e+1), the decrease of the fitted curve itself; close to ``last`` on a
            decaying loss and ~0 on a flat or rising one. Fallback: half the last decrease.
    """
    if len(history) < 3:
        return 0.0
    if p.gain_estimate == "last":
        return max(0.0, float(history[-1]) - predict_next_loss(history))
    popt = _fit_curve(history)
    if popt is None:
        return max(0.0, -0.5 * float(history[-1] - history[-2]))
    e = float(len(history) - 1)
    return max(0.0, float(_exp_curve(e, *popt) - _exp_curve(e + 1.0, *popt)))


def marginal_utility(
    expected_gain: float, u_crit: float, drift: float, p: PolicyParams
) -> float:
    """Delta U(e+1) = delta*G_hat - beta*U_crit/rho - gamma*D (Eq. marginal utility)."""
    return p.delta * expected_gain - p.beta * u_crit / p.rho - p.gamma * drift


def gain_floor(u_crit: float, drift: float, p: PolicyParams) -> float:
    """Smallest predicted gain for which one more epoch is worth it.

    Solving Delta U >= tau_continue for G_hat gives
    g_min = (tau_continue + beta*U_crit/rho + gamma*D) / delta,
    so the threshold rises with safety load and drift.
    """
    return (p.tau_continue + p.beta * u_crit / p.rho + p.gamma * drift) / p.delta


def should_stop(
    expected_gain: float, u_crit: float, drift: float, p: PolicyParams
) -> bool:
    """Stopping rule evaluated after epoch e >= 2.

    legacy  : D > D_max  or  G_hat < G_th  or  Delta U < tau   (submitted runs)
    utility : D > D_max  or  G_hat < g_min(U_crit, D)          (endogenous floor)

    With the paper's weights the legacy utility term can never bind
    (see tests/test_policy.py::test_legacy_utility_test_is_redundant).
    """
    if drift > p.drift_threshold:
        return True
    if p.stop_rule == "utility":
        return expected_gain < gain_floor(u_crit, drift, p)
    return (
        expected_gain < p.gain_threshold
        or marginal_utility(expected_gain, u_crit, drift, p) < p.tau_continue
    )


# --------------------------------------------------------------------------- #
# Stage 3: transmission decision
# --------------------------------------------------------------------------- #
def post_training_decision(
    bandwidth: float,
    final_drift: float,
    actual_gain: float,
    was_in_continue: bool,
    continue_count: int,
    p: PolicyParams,
) -> tuple[str, int, str]:
    """Return (status, new_continue_count, sync_reason)."""
    if bandwidth >= p.bw_threshold:
        return SYNC, 0, SYNC_LINK
    if final_drift > p.drift_sync_fraction * p.drift_threshold:
        return SYNC, 0, SYNC_DRIFT
    if was_in_continue and actual_gain < p.gain_threshold:
        return SYNC, 0, SYNC_GAIN
    new_count = continue_count + 1
    if new_count >= p.max_continue_rounds:
        return SYNC, 0, SYNC_STALENESS
    return CONTINUE, new_count, ""


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
BASELINE_KINDS = ("fixed", "fixedcap", "slackfill", "threshold", "randpart")


def baseline_epochs(kind: str, epochs: int | None, u_crit: float, rho: int) -> int:
    """Local epochs of a baseline in this round (0 means the vehicle skips the round).

    fixed     : E, regardless of the slack (violates constraint (10) when E is large)
    fixedcap  : min(E, slack budget)
    slackfill : slack budget ("run as many epochs as the CPU allows")
    threshold : min(E, slack budget); uploads only on adequate links (see client)
    randpart  : E, for a random fraction of vehicles (see participates)
    """
    budget = slack_epoch_budget(u_crit, rho)
    if kind in ("fixed", "randpart"):
        return int(epochs)
    if kind in ("fixedcap", "threshold"):
        return int(min(int(epochs), budget))
    if kind == "slackfill":
        return budget
    raise ValueError(f"Unknown baseline kind: {kind}")


def derive_seed(*parts: int) -> int:
    """Deterministic 32-bit seed from integer parts (independent of process hash seed)."""
    return int(np.random.SeedSequence([int(x) for x in parts]).generate_state(1)[0])


def participates(seed: int, server_round: int, partition_id: int, prob: float) -> bool:
    """Deterministic Bernoulli(prob) draw per (seed, round, vehicle)."""
    rng = np.random.default_rng(derive_seed(seed, server_round, partition_id, 7919))
    return bool(rng.random() < prob)
