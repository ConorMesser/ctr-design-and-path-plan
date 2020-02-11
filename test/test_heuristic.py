import unittest

from ctrdapp.heuristic.only_goal_distance import OnlyGoalDistance
from ctrdapp.heuristic.square_obstacle_avg_plus_weighted_goal import SquareObstacleAvgPlusWeightedGoal


class TestOnlyGoal(unittest.TestCase):
    def setUp(self) -> None:
        self.gen3 = OnlyGoalDistance(1)
        self.gen2 = OnlyGoalDistance(2)
        self.gen1 = OnlyGoalDistance(3)
        self.gen0 = OnlyGoalDistance(20)

    def test_calculate_cost_from_parent(self):
        self.assertEqual(self.gen3.get_cost(), 1)
        self.assertEqual(self.gen2.get_cost(), 2)
        self.assertEqual(self.gen1.get_cost(), 3)

        self.gen1.calculate_cost_from_parent(self.gen0)
        self.assertEqual(self.gen1.get_cost(), 3)

        self.gen2.calculate_cost_from_parent(self.gen1)
        self.assertEqual(self.gen2.get_cost(), 2)

        self.gen3.calculate_cost_from_parent(self.gen2)
        self.assertEqual(self.gen3.get_cost(), 1)


class TestSOAPWG(unittest.TestCase):
    def setUp(self) -> None:
        self.gen3 = SquareObstacleAvgPlusWeightedGoal(2, 1, -1)
        self.gen2 = SquareObstacleAvgPlusWeightedGoal(2, 5, 10)
        self.gen1 = SquareObstacleAvgPlusWeightedGoal(2, 4, 20)
        self.gen0 = SquareObstacleAvgPlusWeightedGoal(2, 50, 1)

    def test_calculate_cost_from_parent(self):
        self.assertEqual(self.gen3.get_cost(), -1)
        self.assertEqual(self.gen2.get_cost(), 25)
        self.assertEqual(self.gen1.get_cost(), 44)

        self.gen1.calculate_cost_from_parent(self.gen0)
        self.assertEqual(self.gen1.get_cost(), 40 + 1/16)
        self.assertEqual(self.gen1.generation, 1)
        self.assertEqual(self.gen1.avg_obstacle_min, 1/16)

        self.gen2.calculate_cost_from_parent(self.gen1)
        self.assertEqual(self.gen2.get_cost(), 20.05125)
        self.assertEqual(self.gen2.generation, 2)
        self.assertAlmostEqual(self.gen2.avg_obstacle_min, 0.05125)

        self.gen3.calculate_cost_from_parent(self.gen2)
        self.assertEqual(self.gen3.get_cost(), -1.6325)
        self.assertEqual(self.gen3.generation, 3)
        self.assertAlmostEqual(self.gen3.avg_obstacle_min, 0.3675)


if __name__ == '__main__':
    unittest.main()
