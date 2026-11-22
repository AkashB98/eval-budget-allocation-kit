"""Seeded synthetic scripted-agent scores (no LLM, no network).

SYNTHETIC DATA ONLY. A deterministic scripted agent is simulated with a
nested random-effects model (generalizability-theory style):

    score(q, t, r) = mu + question_effect[q]
                     + trajectory_noise[q, t]
                     + grader_noise[q, t, r]

question_effect  ~ N(0, sigma_q)   -- some questions are harder/better
trajectory_noise ~ N(0, sigma_t)   -- same question, different run
grader_noise     ~ N(0, sigma_g)   -- same trajectory, re-graded

Grader noise is a real simulated level: sigma_g defaults to a non-zero
value in every regime and is never silently zero (see REGIMES).
"""
from __future__ import annotations

import random

# True (planted) variance components per regime. Variances, not SDs.
REGIMES = {
    # Low grader noise: repeats buy little; questions/trajectories win.
    "standard": {"mu": 0.70, "var_q": 0.04, "var_t": 0.09, "var_g": 0.02},
    # High grader noise: re-grading the same trajectory genuinely helps.
    "high_grader_noise": {"mu": 0.70, "var_q": 0.04, "var_t": 0.09, "var_g": 0.25},
}

DEFAULT_SEED = 42
HELDOUT_SEED = 20240517
POOL_SIZE = 300


def _check_dims(n_questions, n_traj, n_repeats):
    if n_questions < 1 or n_traj < 1 or n_repeats < 1:
        raise ValueError("allocation dimensions must all be >= 1")
    if n_questions > POOL_SIZE:
        raise ValueError(f"question pool exhausted: {n_questions} > {POOL_SIZE}")


def generate_scores(n_questions, n_traj, n_repeats, regime="standard",
                    seed=DEFAULT_SEED):
    """Return nested scores: scores[q][t] is a list of R re-grade scores."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime: {regime}")
    _check_dims(n_questions, n_traj, n_repeats)
    p = REGIMES[regime]
    rng = random.Random(seed)
    sd_q, sd_t, sd_g = p["var_q"] ** 0.5, p["var_t"] ** 0.5, p["var_g"] ** 0.5
    out = []
    for _q in range(n_questions):
        q_eff = rng.gauss(0.0, sd_q)
        trajs = []
        for _t in range(n_traj):
            t_noise = rng.gauss(0.0, sd_t)
            base = p["mu"] + q_eff + t_noise
            trajs.append([base + rng.gauss(0.0, sd_g) for _r in range(n_repeats)])
        out.append(trajs)
    return out


def generate_pilot(n_questions=100, n_traj=3, n_repeats=3,
                   regime="standard", seed=DEFAULT_SEED):
    """A pilot run: a modest nested sample used to estimate components."""
    return generate_scores(n_questions, n_traj, n_repeats,
                           regime=regime, seed=seed)


def estimate_mean(scores):
    """Grand mean of a nested score structure (the eval's headline number)."""
    total, n = 0.0, 0
    for trajs in scores:
        for reps in trajs:
            total += sum(reps)
            n += len(reps)
    if n == 0:
        raise ValueError("no scores")
    return total / n


def simulate_mean(n_questions, n_traj, n_repeats, regime="standard",
                  seed=0):
    """One simulated eval run's estimated mean under an allocation.

    Uses the aggregated form of the same nested model (mathematically
    identical distribution for the grand mean, much faster than drawing
    every re-grade): the mean of Q question effects has variance
    var_q/Q, and so on down the nesting.
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime: {regime}")
    _check_dims(n_questions, n_traj, n_repeats)
    p = REGIMES[regime]
    rng = random.Random(seed)
    return (p["mu"]
            + rng.gauss(0.0, (p["var_q"] / n_questions) ** 0.5)
            + rng.gauss(0.0, (p["var_t"] / (n_questions * n_traj)) ** 0.5)
            + rng.gauss(0.0, (p["var_g"] / (n_questions * n_traj * n_repeats)) ** 0.5))


def empirical_se(n_questions, n_traj, n_repeats, regime="standard",
                 reps=150, seed=HELDOUT_SEED):
    """Empirical SE of the estimated mean over `reps` held-out simulations."""
    if reps < 2:
        raise ValueError("reps must be >= 2")
    means = [simulate_mean(n_questions, n_traj, n_repeats, regime=regime,
                           seed=seed + 1000003 * i) for i in range(reps)]
    m = sum(means) / len(means)
    var = sum((x - m) ** 2 for x in means) / (len(means) - 1)
    return var ** 0.5
