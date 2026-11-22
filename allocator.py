"""Variance components, cost accounting, and budget allocation.

Textbook two-stage / generalizability-theory optimal allocation: given
variance components estimated from a pilot, choose (Q, T, R) -- questions,
trajectories per question, re-grades per trajectory -- minimising the
standard error of the estimated mean score subject to a token budget,
a finite question pool, and unequal per-level costs.
"""
from __future__ import annotations

from generator import POOL_SIZE

# Unequal costs (tokens). A rollout (new question or extra trajectory)
# pays for generation; a repeat re-grade pays for grading only.
COST_ROLLOUT = 1000
COST_REGRADE = 100
DEFAULT_BUDGET = 200000
MAX_T = 10
MAX_R = 5


def allocation_cost(n_questions, n_traj, n_repeats,
                    cost_rollout=COST_ROLLOUT, cost_regrade=COST_REGRADE):
    """Exact token cost of an allocation.

    Every trajectory costs one rollout (its first grade is included);
    each additional re-grade of that trajectory costs cost_regrade.
    """
    if n_questions < 1 or n_traj < 1 or n_repeats < 1:
        raise ValueError("allocation dimensions must all be >= 1")
    return (n_questions * n_traj * cost_rollout
            + n_questions * n_traj * (n_repeats - 1) * cost_regrade)


def predicted_variance(var_q, var_t, var_g, n_questions, n_traj, n_repeats):
    """Variance of the estimated mean under the nested model."""
    return (var_q / n_questions
            + var_t / (n_questions * n_traj)
            + var_g / (n_questions * n_traj * n_repeats))


def predicted_se(var_q, var_t, var_g, n_questions, n_traj, n_repeats):
    return predicted_variance(var_q, var_t, var_g,
                              n_questions, n_traj, n_repeats) ** 0.5


def estimate_variance_components(scores):
    """Method-of-moments variance components from nested pilot scores.

    scores[q][t] = list of R re-grade scores. Unbiased nested ANOVA-style
    decomposition, floored at 0 (a noisy tiny pilot can go negative):
      var_g: pooled within-trajectory variance of re-grades
      var_t: within-question variance of trajectory means, minus var_g/R
      var_q: variance of question means, minus (var_t + var_g/R)/T
    """
    if not scores or not scores[0] or not scores[0][0]:
        raise ValueError("empty pilot scores")
    n_q = len(scores)
    n_t = len(scores[0])
    n_r = len(scores[0][0])
    if any(len(t) != n_t or any(len(r) != n_r for r in t) for t in scores):
        raise ValueError("pilot scores must be a balanced nested design")
    if n_q < 2 or n_t < 2 or n_r < 2:
        raise ValueError("pilot needs >= 2 questions, trajectories and repeats "
                         "to estimate all three components")

    # var_g: pooled within-(q,t) variance.
    ss_g, df_g = 0.0, 0
    traj_means = []  # traj_means[q][t]
    for trajs in scores:
        row = []
        for reps in trajs:
            m = sum(reps) / n_r
            row.append(m)
            ss_g += sum((x - m) ** 2 for x in reps)
            df_g += n_r - 1
        traj_means.append(row)
    var_g = ss_g / df_g

    # var_t: within-question variance of trajectory means.
    ss_t, df_t = 0.0, 0
    q_means = []
    for row in traj_means:
        m = sum(row) / n_t
        q_means.append(m)
        ss_t += sum((x - m) ** 2 for x in row)
        df_t += n_t - 1
    var_t = max(0.0, ss_t / df_t - var_g / n_r)

    # var_q: variance of question means.
    grand = sum(q_means) / n_q
    var_q_obs = sum((x - grand) ** 2 for x in q_means) / (n_q - 1)
    var_q = max(0.0, var_q_obs - (var_t + var_g / n_r) / n_t)
    return {"var_q": var_q, "var_t": var_t, "var_g": var_g}


def _max_questions(n_traj, n_repeats, budget, pool=POOL_SIZE):
    per_q = allocation_cost(1, n_traj, n_repeats)
    return min(pool, budget // per_q)


def search_allocation(var_q, var_t, var_g, budget=DEFAULT_BUDGET,
                      pool=POOL_SIZE, max_t=MAX_T, max_r=MAX_R):
    """Exhaustive grid search for the SE-minimising feasible allocation."""
    if budget < COST_ROLLOUT:
        raise ValueError("budget too small for a single rollout")
    best = None
    for t in range(1, max_t + 1):
        for r in range(1, max_r + 1):
            q = _max_questions(t, r, budget, pool)
            if q < 1:
                continue
            cost = allocation_cost(q, t, r)
            se = predicted_se(var_q, var_t, var_g, q, t, r)
            key = (round(se, 12), cost, -q, t, r)
            if best is None or key < best[0]:
                best = (key, {"n_questions": q, "n_traj": t, "n_repeats": r,
                              "cost": cost, "predicted_se": se})
    if best is None:
        raise ValueError("no feasible allocation under budget")
    return best[1]


def scaled_allocation(n_traj, n_repeats, budget=DEFAULT_BUDGET,
                      pool=POOL_SIZE):
    """Largest feasible allocation with a fixed (T, R) shape (a baseline)."""
    q = _max_questions(n_traj, n_repeats, budget, pool)
    if q < 1:
        raise ValueError("no feasible allocation under budget")
    return {"n_questions": q, "n_traj": n_traj, "n_repeats": n_repeats,
            "cost": allocation_cost(q, n_traj, n_repeats)}


# Naive baselines, each scaled to spend (nearly) the same budget.
BASELINE_SHAPES = {
    "equal_split": (2, 2),        # balanced T and R, fewer questions
    "repeats_heavy": (1, 5),      # 5-reads style: re-grade everything
    "trajectories_heavy": (5, 1),  # many runs per question, one grade
}


def baseline_allocations(budget=DEFAULT_BUDGET, pool=POOL_SIZE):
    return {name: scaled_allocation(t, r, budget, pool)
            for name, (t, r) in BASELINE_SHAPES.items()}
