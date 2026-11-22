import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from allocator import (allocation_cost, baseline_allocations,
                       estimate_variance_components, predicted_se,
                       scaled_allocation, search_allocation)
from generator import POOL_SIZE, generate_pilot, generate_scores


class TestCostAccounting(unittest.TestCase):
    def test_single_rollout(self):
        self.assertEqual(allocation_cost(1, 1, 1), 1000)

    def test_regrades_are_cheap(self):
        # 10 q x 1 t x 5 r = 10 rollouts + 40 regrades
        self.assertEqual(allocation_cost(10, 1, 5), 10 * 1000 + 40 * 100)

    def test_trajectories_cost_rollouts(self):
        self.assertEqual(allocation_cost(10, 3, 1), 30 * 1000)

    def test_mixed_exact(self):
        # 7 q x 2 t x 3 r = 14 rollouts + 28 regrades
        self.assertEqual(allocation_cost(7, 2, 3), 14 * 1000 + 28 * 100)

    def test_invalid_dims(self):
        with self.assertRaises(ValueError):
            allocation_cost(0, 1, 1)


class TestEstimation(unittest.TestCase):
    def test_components_recovered_large_pilot(self):
        scores = generate_pilot(300, 4, 4, seed=123)
        est = estimate_variance_components(scores)
        self.assertAlmostEqual(est["var_q"], 0.04, delta=0.03)
        self.assertAlmostEqual(est["var_t"], 0.09, delta=0.04)
        self.assertAlmostEqual(est["var_g"], 0.02, delta=0.01)

    def test_grader_noise_nonzero_and_regime_ordered(self):
        std = estimate_variance_components(generate_pilot(200, 3, 3, seed=7))
        high = estimate_variance_components(
            generate_pilot(200, 3, 3, regime="high_grader_noise", seed=7))
        self.assertGreater(std["var_g"], 0.0)
        self.assertGreater(high["var_g"], std["var_g"] * 3)

    def test_unbalanced_rejected(self):
        scores = generate_scores(5, 2, 2, seed=1)
        scores[0] = scores[0][:1]
        with self.assertRaises(ValueError):
            estimate_variance_components(scores)

    def test_too_small_rejected(self):
        with self.assertRaises(ValueError):
            estimate_variance_components(generate_scores(5, 1, 2, seed=1))


class TestSearch(unittest.TestCase):
    def test_feasible_and_within_pool(self):
        for budget in (5000, 60000, 200000, 500000, 2000000):
            a = search_allocation(0.04, 0.09, 0.02, budget=budget)
            self.assertLessEqual(a["cost"], budget)
            self.assertLessEqual(a["n_questions"], POOL_SIZE)
            self.assertEqual(a["cost"], allocation_cost(
                a["n_questions"], a["n_traj"], a["n_repeats"]))

    def test_pool_cap_binds(self):
        a = search_allocation(0.04, 0.09, 0.02, budget=2000000)
        self.assertEqual(a["n_questions"], POOL_SIZE)
        self.assertGreater(a["n_traj"] * a["n_repeats"], 1)

    def test_high_grader_noise_prefers_repeats(self):
        low = search_allocation(0.04, 0.09, 0.02, budget=200000)
        high = search_allocation(0.04, 0.09, 0.25, budget=200000)
        self.assertEqual(low["n_repeats"], 1)
        self.assertGreater(high["n_repeats"], 1)

    def test_predicted_se_formula(self):
        se = predicted_se(0.04, 0.09, 0.02, 100, 2, 2)
        expect = (0.04 / 100 + 0.09 / 200 + 0.02 / 400) ** 0.5
        self.assertAlmostEqual(se, expect)

    def test_grader_noise_raises_se(self):
        self.assertGreater(predicted_se(0.04, 0.09, 0.25, 50, 2, 1),
                           predicted_se(0.04, 0.09, 0.02, 50, 2, 1))

    def test_repeats_reduce_grader_term(self):
        self.assertLess(predicted_se(0.04, 0.09, 0.25, 50, 1, 5),
                        predicted_se(0.04, 0.09, 0.25, 50, 1, 1))

    def test_baselines_feasible(self):
        for a in baseline_allocations(200000).values():
            self.assertLessEqual(a["cost"], 200000)
            self.assertLessEqual(a["n_questions"], POOL_SIZE)

    def test_scaled_allocation_exact(self):
        a = scaled_allocation(1, 5, budget=200000)
        self.assertEqual(a["n_questions"], 200000 // 1400)
        self.assertEqual(a["cost"], allocation_cost(a["n_questions"], 1, 5))

    def test_tiny_budget_rejected(self):
        with self.assertRaises(ValueError):
            search_allocation(0.04, 0.09, 0.02, budget=500)


if __name__ == "__main__":
    unittest.main()
