"""Full report: pilot -> components -> allocation -> held-out verification."""
from __future__ import annotations

from allocator import (DEFAULT_BUDGET, baseline_allocations,
                       estimate_variance_components, predicted_se,
                       search_allocation)
from generator import (HELDOUT_SEED, REGIMES, empirical_se, generate_pilot)

PILOT_Q, PILOT_T, PILOT_R = 100, 3, 3
SMALL_PILOT_Q, LARGE_PILOT_Q = 20, 200
DEGRADATION_REPS = 25
SIM_REPS = 150


def _with_ses(alloc, true_params, est_params, regime, reps=SIM_REPS):
    out = dict(alloc)
    out["predicted_se"] = round(predicted_se(
        est_params["var_q"], est_params["var_t"], est_params["var_g"],
        alloc["n_questions"], alloc["n_traj"], alloc["n_repeats"]), 6)
    out["true_se"] = round(predicted_se(
        true_params["var_q"], true_params["var_t"], true_params["var_g"],
        alloc["n_questions"], alloc["n_traj"], alloc["n_repeats"]), 6)
    out["empirical_se"] = round(empirical_se(
        alloc["n_questions"], alloc["n_traj"], alloc["n_repeats"],
        regime=regime, reps=reps, seed=HELDOUT_SEED), 6)
    return out


def pilot_degradation(budget=DEFAULT_BUDGET, regime="standard"):
    """Mean true SE of the pilot-recommended allocation, small vs large pilot.

    A tiny pilot mis-estimates the components, so its recommendation is
    worse on held-out truth. Averaged over DEGRADATION_REPS pilot seeds so
    the comparison is a property of pilot size, not one lucky draw.
    """
    true_p = REGIMES[regime]
    result = {}
    for label, n_q, base_seed in (("small_pilot", SMALL_PILOT_Q, 7000),
                                  ("large_pilot", LARGE_PILOT_Q, 9000)):
        ses, errs = [], []
        for i in range(DEGRADATION_REPS):
            pilot = generate_pilot(n_q, PILOT_T, PILOT_R, regime=regime,
                                   seed=base_seed + i)
            est = estimate_variance_components(pilot)
            alloc = search_allocation(est["var_q"], est["var_t"], est["var_g"],
                                      budget=budget)
            ses.append(predicted_se(true_p["var_q"], true_p["var_t"],
                                    true_p["var_g"], alloc["n_questions"],
                                    alloc["n_traj"], alloc["n_repeats"]))
            errs.append(sum(abs(est[k] - true_p[k]) / true_p[k]
                            for k in ("var_q", "var_t", "var_g")) / 3)
        result[label] = {
            "pilot_questions": n_q,
            "mean_achieved_true_se": round(sum(ses) / len(ses), 6),
            "worst_achieved_true_se": round(max(ses), 6),
            "mean_component_relative_error": round(sum(errs) / len(errs), 6)}
    result["se_inflation_small_vs_large"] = round(
        result["small_pilot"]["mean_achieved_true_se"]
        / result["large_pilot"]["mean_achieved_true_se"], 6)
    return result


def build_report(budget=DEFAULT_BUDGET, regime="standard", seed=42):
    if regime not in REGIMES:
        raise ValueError(f"unknown regime: {regime}")
    if budget < 1000:
        raise ValueError("budget too small for a single rollout")
    true_p = REGIMES[regime]
    pilot = generate_pilot(PILOT_Q, PILOT_T, PILOT_R, regime=regime, seed=seed)
    est = estimate_variance_components(pilot)
    est_r = {k: round(v, 6) for k, v in est.items()}

    rec_raw = search_allocation(est["var_q"], est["var_t"], est["var_g"],
                                budget=budget)
    recommended = _with_ses(rec_raw, true_p, est, regime)
    baselines = {name: _with_ses(a, true_p, est, regime)
                 for name, a in baseline_allocations(budget).items()}

    # Second regime: same machinery, high grader noise -> repeats pay.
    hg = "high_grader_noise"
    hg_pilot = generate_pilot(PILOT_Q, PILOT_T, PILOT_R, regime=hg, seed=seed)
    hg_est = estimate_variance_components(hg_pilot)
    hg_rec_raw = search_allocation(hg_est["var_q"], hg_est["var_t"],
                                   hg_est["var_g"], budget=budget)
    hg_recommended = _with_ses(hg_rec_raw, REGIMES[hg], hg_est, hg)

    beats = all(recommended["empirical_se"] < b["empirical_se"]
                for b in baselines.values())
    return {
        "regime": regime,
        "budget": budget,
        "pool_size": 300,
        "costs": {"rollout": 1000, "regrade": 100},
        "true_components": {"var_q": true_p["var_q"], "var_t": true_p["var_t"],
                            "var_g": true_p["var_g"]},
        "pilot": {"n_questions": PILOT_Q, "n_traj": PILOT_T,
                  "n_repeats": PILOT_R, "seed": seed,
                  "estimated_components": est_r},
        "recommended": recommended,
        "baselines": baselines,
        "recommended_beats_all_baselines_empirical": beats,
        "high_grader_noise_regime": {
            "true_components": {"var_q": REGIMES[hg]["var_q"],
                                "var_t": REGIMES[hg]["var_t"],
                                "var_g": REGIMES[hg]["var_g"]},
            "recommended": hg_recommended,
            "repeats_help": hg_recommended["n_repeats"] > 1,
        },
        "pilot_degradation": pilot_degradation(budget=budget, regime=regime),
        "heldout": {"seed": HELDOUT_SEED, "sim_reps": SIM_REPS},
    }
