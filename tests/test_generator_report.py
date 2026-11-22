import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import main as cli_main
from generator import (POOL_SIZE, empirical_se, estimate_mean,
                       generate_scores, simulate_mean)
from report import build_report


class TestGenerator(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(generate_scores(10, 2, 2, seed=3),
                         generate_scores(10, 2, 2, seed=3))

    def test_shape(self):
        s = generate_scores(7, 3, 4, seed=3)
        self.assertEqual((len(s), len(s[0]), len(s[0][0])), (7, 3, 4))

    def test_pool_limit(self):
        with self.assertRaises(ValueError):
            generate_scores(POOL_SIZE + 1, 1, 1)

    def test_mean_near_mu(self):
        s = generate_scores(300, 3, 3, seed=11)
        self.assertAlmostEqual(estimate_mean(s), 0.70, delta=0.05)

    def test_empirical_se_shrinks_with_questions(self):
        small = empirical_se(25, 1, 1, reps=60, seed=5)
        large = empirical_se(200, 1, 1, reps=60, seed=5)
        self.assertLess(large, small)

    def test_simulate_mean_deterministic(self):
        self.assertEqual(simulate_mean(50, 2, 2, seed=9),
                         simulate_mean(50, 2, 2, seed=9))


class TestReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_determinism_same_seed_same_report(self):
        again = build_report()
        self.assertEqual(json.dumps(self.report, sort_keys=True),
                         json.dumps(again, sort_keys=True))

    def test_recommended_beats_baselines(self):
        rec = self.report["recommended"]
        for base in self.report["baselines"].values():
            self.assertLess(rec["empirical_se"], base["empirical_se"])
        self.assertTrue(self.report["recommended_beats_all_baselines_empirical"])

    def test_recommended_feasible(self):
        rec = self.report["recommended"]
        self.assertLessEqual(rec["cost"], self.report["budget"])
        self.assertLessEqual(rec["n_questions"], POOL_SIZE)

    def test_high_grader_noise_repeats_help(self):
        hg = self.report["high_grader_noise_regime"]
        self.assertTrue(hg["repeats_help"])
        self.assertGreater(hg["recommended"]["n_repeats"], 1)

    def test_pilot_degradation(self):
        deg = self.report["pilot_degradation"]
        self.assertGreater(deg["small_pilot"]["mean_achieved_true_se"],
                           deg["large_pilot"]["mean_achieved_true_se"])
        self.assertGreater(deg["se_inflation_small_vs_large"], 1.0)


class TestCli(unittest.TestCase):
    def test_check_passes(self):
        self.assertEqual(cli_main(["check"]), 0)

    def test_check_bad_input_file(self):
        self.assertEqual(cli_main(["check", "--input", "/tmp/nope-xyz.json"]), 2)

    def test_check_bad_budget(self):
        self.assertEqual(cli_main(["check", "--budget", "10"]), 2)


if __name__ == "__main__":
    unittest.main()
