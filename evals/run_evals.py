"""Golden evals for eval-budget-allocation-kit.

Deterministic: eval_report.json is byte-identical across runs
(no timestamps, no temp paths). Run: python3 evals/run_evals.py
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from allocator import (allocation_cost,  # noqa: E402
                       estimate_variance_components, search_allocation)
from cli import main as cli_main  # noqa: E402
from generator import generate_pilot  # noqa: E402
from report import build_report  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = []


def check(name, passed, detail=""):
    RESULTS.append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
          + (f" - {detail}" if detail else ""))
    return passed


def _run_cli(argv):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return cli_main(argv)


def main():
    print("== golden evals: eval-budget-allocation-kit ==")
    report = build_report()
    rec = report["recommended"]
    bases = report["baselines"]

    check("G1 recommended allocation is feasible (budget + pool + exact cost)",
          rec["cost"] <= report["budget"] and rec["n_questions"] <= 300
          and rec["cost"] == allocation_cost(rec["n_questions"],
                                             rec["n_traj"], rec["n_repeats"]),
          f"Q={rec['n_questions']} T={rec['n_traj']} R={rec['n_repeats']} "
          f"cost={rec['cost']} budget={report['budget']}")

    check("G2 recommended beats all naive baselines on held-out empirical SE",
          report["recommended_beats_all_baselines_empirical"] is True
          and all(rec["empirical_se"] < b["empirical_se"] for b in bases.values()),
          f"rec={rec['empirical_se']} " + " ".join(
              f"{k}={v['empirical_se']}" for k, v in sorted(bases.items())))

    check("G3 empirical SE tracks analytic true SE for the recommended allocation",
          abs(rec["empirical_se"] - rec["true_se"]) < 0.25 * rec["true_se"],
          f"empirical={rec['empirical_se']} true={rec['true_se']}")

    est = report["pilot"]["estimated_components"]
    check("G4 pilot recovers planted variance components (incl. non-zero grader noise)",
          abs(est["var_q"] - 0.04) < 0.03 and abs(est["var_t"] - 0.09) < 0.05
          and 0.005 < est["var_g"] < 0.06,
          f"est={est} true={report['true_components']}")

    hg = report["high_grader_noise_regime"]["recommended"]
    check("G5 high grader-noise regime: repeats DO help (R > 1 recommended)",
          report["high_grader_noise_regime"]["repeats_help"] is True
          and hg["n_repeats"] > 1,
          f"Q={hg['n_questions']} T={hg['n_traj']} R={hg['n_repeats']} "
          f"empirical_se={hg['empirical_se']}")

    deg = report["pilot_degradation"]
    check("G6 small pilot degrades the achieved SE vs large pilot",
          deg["small_pilot"]["mean_achieved_true_se"]
          > deg["large_pilot"]["mean_achieved_true_se"]
          and deg["se_inflation_small_vs_large"] > 1.0
          and deg["small_pilot"]["worst_achieved_true_se"]
          > deg["large_pilot"]["worst_achieved_true_se"]
          and deg["small_pilot"]["mean_component_relative_error"]
          > 1.5 * deg["large_pilot"]["mean_component_relative_error"],
          f"small_mean={deg['small_pilot']['mean_achieved_true_se']} "
          f"large_mean={deg['large_pilot']['mean_achieved_true_se']} "
          f"small_worst={deg['small_pilot']['worst_achieved_true_se']} "
          f"large_worst={deg['large_pilot']['worst_achieved_true_se']} "
          f"rel_err small={deg['small_pilot']['mean_component_relative_error']} "
          f"large={deg['large_pilot']['mean_component_relative_error']}")

    capped = search_allocation(0.04, 0.09, 0.02, budget=2000000)
    check("G7 finite pool binds: huge budget caps Q at the pool and spends on T/R",
          capped["n_questions"] == 300
          and capped["n_traj"] * capped["n_repeats"] > 1
          and capped["cost"] <= 2000000,
          f"Q={capped['n_questions']} T={capped['n_traj']} R={capped['n_repeats']} "
          f"cost={capped['cost']}")

    codes = (_run_cli(["check"]), _run_cli(["check", "--budget", "10"]),
             _run_cli(["check", "--input", "/tmp/no-such-eval-budget-xyz.json"]))
    check("G8 CLI check exit codes 0 / 2 / 2 (pass / bad budget / missing file)",
          codes == (0, 2, 2), f"codes={codes}")

    again = build_report()
    check("G9 full report is deterministic (same seed -> same report)",
          json.dumps(report, sort_keys=True) == json.dumps(again, sort_keys=True),
          "two build_report() calls byte-identical as JSON")

    est2 = estimate_variance_components(
        generate_pilot(200, 3, 3, regime="high_grader_noise", seed=7))
    est1 = estimate_variance_components(generate_pilot(200, 3, 3, seed=7))
    check("G10 grader noise is a real level: high-noise pilot estimates >> standard",
          est2["var_g"] > 3 * est1["var_g"] and est1["var_g"] > 0,
          f"standard var_g={round(est1['var_g'], 6)} high var_g={round(est2['var_g'], 6)}")

    passed = sum(1 for r in RESULTS if r["passed"])
    payload = {"total": len(RESULTS), "passed": passed,
               "failed": len(RESULTS) - passed, "results": RESULTS,
               "headline": {
                   "recommended": {k: rec[k] for k in (
                       "n_questions", "n_traj", "n_repeats", "cost",
                       "predicted_se", "true_se", "empirical_se")},
                   "baselines_empirical_se": {
                       k: v["empirical_se"] for k, v in sorted(bases.items())},
                   "high_grader_noise_recommended": {k: hg[k] for k in (
                       "n_questions", "n_traj", "n_repeats", "empirical_se")},
                   "pilot_degradation": deg,
                   "estimated_components": est,
               }}
    with open(os.path.join(HERE, "eval_report.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\n{passed}/{len(RESULTS)} golden evals passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
