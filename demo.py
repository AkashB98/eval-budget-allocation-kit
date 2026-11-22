"""End-to-end demo: pilot -> components -> allocation -> held-out check.

SYNTHETIC DATA ONLY. Runs fully offline; every stage asserts.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from allocator import allocation_cost  # noqa: E402
from cli import EXIT_OK, main as cli_main  # noqa: E402
from generator import POOL_SIZE  # noqa: E402
from report import build_report  # noqa: E402


def run():
    print("== eval-budget-allocation-kit demo ==")
    report = build_report()
    print(f"\n== 1. pilot ({report['pilot']['n_questions']} questions x "
          f"{report['pilot']['n_traj']} trajectories x "
          f"{report['pilot']['n_repeats']} re-grades) ==")
    print(f"   true components:      {report['true_components']}")
    print(f"   estimated components: {report['pilot']['estimated_components']}")

    rec = report["recommended"]
    print(f"\n== 2. recommended allocation under budget {report['budget']} ==")
    print(f"   Q={rec['n_questions']} T={rec['n_traj']} R={rec['n_repeats']} "
          f"cost={rec['cost']} predicted_se={rec['predicted_se']} "
          f"true_se={rec['true_se']} empirical_se={rec['empirical_se']}")
    assert rec["cost"] == allocation_cost(rec["n_questions"], rec["n_traj"],
                                          rec["n_repeats"])
    assert rec["cost"] <= report["budget"]
    assert rec["n_questions"] <= POOL_SIZE

    print("\n== 3. naive baselines at the same budget (held-out empirical SE) ==")
    for name, base in sorted(report["baselines"].items()):
        print(f"   {name}: Q={base['n_questions']} T={base['n_traj']} "
              f"R={base['n_repeats']} cost={base['cost']} "
              f"empirical_se={base['empirical_se']}")
        assert base["cost"] <= report["budget"]
        assert rec["empirical_se"] < base["empirical_se"], name
    assert report["recommended_beats_all_baselines_empirical"] is True

    hg = report["high_grader_noise_regime"]["recommended"]
    print("\n== 4. high grader-noise regime: repeats DO help ==")
    print(f"   recommended Q={hg['n_questions']} T={hg['n_traj']} "
          f"R={hg['n_repeats']} empirical_se={hg['empirical_se']}")
    assert hg["n_repeats"] > 1

    deg = report["pilot_degradation"]
    print("\n== 5. small-pilot degradation ==")
    print(f"   small pilot (20 q):  mean achieved true SE "
          f"{deg['small_pilot']['mean_achieved_true_se']}, worst "
          f"{deg['small_pilot']['worst_achieved_true_se']}, component "
          f"rel. error {deg['small_pilot']['mean_component_relative_error']}")
    print(f"   large pilot (200 q): mean achieved true SE "
          f"{deg['large_pilot']['mean_achieved_true_se']}, worst "
          f"{deg['large_pilot']['worst_achieved_true_se']}, component "
          f"rel. error {deg['large_pilot']['mean_component_relative_error']}")
    print(f"   mean-SE inflation: {deg['se_inflation_small_vs_large']}x "
          f"(the SE surface is flat near the optimum; the visible damage "
          f"is component error and worst-case SE)")
    assert deg["small_pilot"]["mean_achieved_true_se"] > \
        deg["large_pilot"]["mean_achieved_true_se"]
    assert deg["small_pilot"]["mean_component_relative_error"] > \
        deg["large_pilot"]["mean_component_relative_error"]

    print("\n== 6. CLI check gate ==")
    code = cli_main(["check"])
    print(f"   check exit={code}")
    assert code == EXIT_OK
    print("\nDemo OK: pilot, allocation, baselines, grader-noise regime, "
          "pilot degradation and check gate verified.")
    return 0


if __name__ == "__main__":
    sys.exit(run())
