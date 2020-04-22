import unittest

from ctrdapp.heuristic.only_goal_distance import OnlyGoalDistance
from ctrdapp.heuristic.square_obstacle_avg_plus_weighted_goal import SquareObstacleAvgPlusWeightedGoal
from ctrdapp.heuristic.follow_the_leader import FollowTheLeaderWInsertion, FollowTheLeader


class TestOnlyGoal(unittest.TestCase):
    def setUp(self) -> None:
        self.gen4 = OnlyGoalDistance(1)
        self.gen3 = OnlyGoalDistance(2)
        self.gen2 = OnlyGoalDistance(3)
        self.gen1 = OnlyGoalDistance(10)

    def test_calculate_cost_from_parent(self):
        self.assertEqual(self.gen4.get_cost(), 1)
        self.assertEqual(self.gen3.get_cost(), 2)
        self.assertEqual(self.gen2.get_cost(), 3)

        self.gen2.calculate_cost_from_parent(self.gen1)
        self.assertEqual(self.gen2.get_cost(), 3)

        self.gen3.calculate_cost_from_parent(self.gen2)
        self.assertEqual(self.gen3.get_cost(), 2)

        self.gen4.calculate_cost_from_parent(self.gen3)
        self.assertEqual(self.gen4.get_cost(), 1)


class TestSOAPWG(unittest.TestCase):
    def setUp(self) -> None:
        self.gen3 = SquareObstacleAvgPlusWeightedGoal(2, 1, -1)
        self.gen2 = SquareObstacleAvgPlusWeightedGoal(2, 5, 10)
        self.gen1 = SquareObstacleAvgPlusWeightedGoal(2, 4, 20)
        self.gen0 = SquareObstacleAvgPlusWeightedGoal(2, 300, 1)

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


class TestFTL(unittest.TestCase):
    def setUp(self) -> None:
        self.ftl_0 = FollowTheLeader(True, [[[0, 0, 0, 100, 0, 0], [0, 0, 0, 100, 0, 0]]])
        self.ftl_1 = FollowTheLeader(True, [[[0, 0, 0, 2, 2, 1], [4, 0, 3, 0, 0, 0]]])
        self.ftl_2 = FollowTheLeader(True, [[[0, 0, 0, 2, 2, 1], [2, 1, 2, 0, 5, 10]]])
        self.ftl_3 = FollowTheLeader(True, [[[6, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]])

        self.ftl_1_full = FollowTheLeader(False, [[[0, 0, 0, 2, 2, 1], [4, 0, 3, 0, 0, 0]]])
        self.ftl_2_full = FollowTheLeader(False, [[[0, 0, 0, 2, 2, 1], [2, 1, 2, 0, 5, 10]]])
        self.ftl_3_full = FollowTheLeader(False, [[[6, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]]])

    # test only_tip vs full (average)
    # test magnitudes (omega vs. velocity)
    # test parent averages
    def testMagnitude(self):
        self.assertEqual(self.ftl_0.get_cost(), 100)
        self.assertEqual(self.ftl_1.get_cost(), 5)
        self.assertEqual(self.ftl_2.get_cost(), 3)
        self.assertEqual(self.ftl_3.get_cost(), 1)

    def testParent(self):
        self.ftl_1.calculate_cost_from_parent(self.ftl_0)
        self.ftl_2.calculate_cost_from_parent(self.ftl_1)
        self.ftl_3.calculate_cost_from_parent(self.ftl_2)

        self.assertEqual(self.ftl_1.get_cost(), 5)
        self.assertEqual(self.ftl_2.get_cost(), 4)
        self.assertEqual(self.ftl_3.get_cost(), 3)

    def testTipVsFull(self):
        self.assertEqual(self.ftl_1_full.get_cost(), 4)
        self.assertEqual(self.ftl_2_full.get_cost(), 3)
        self.assertEqual(self.ftl_3_full.get_cost(), 8/3)

        self.ftl_1_full.calculate_cost_from_parent(self.ftl_0)
        self.ftl_2_full.calculate_cost_from_parent(self.ftl_1_full)
        self.ftl_3_full.calculate_cost_from_parent(self.ftl_2_full)

        self.assertEqual(self.ftl_1_full.get_cost(), 4)  # parent generation is 0 - doesn't affect average calculation
        self.assertEqual(self.ftl_2_full.get_cost(), 7/2)
        self.assertAlmostEqual(self.ftl_3_full.get_cost(), 29/9)


class TestFTLInsertion(unittest.TestCase):
    def setUp(self) -> None:
        self.ftl_length_0 = FollowTheLeaderWInsertion(True, 10, [[[2, 0, 0, 0, 0, 0]]], 1.5)
        self.ftl_length_1 = FollowTheLeaderWInsertion(True, 10, [[[4, 0, 0, 0, 0, 0]]], 1)
        self.ftl_length_2 = FollowTheLeaderWInsertion(True, 10, [[[6, 0, 0, 0, 0, 0]]], 0.5)

    def testCost(self):
        self.assertEqual(self.ftl_length_0.get_cost(), 17)
        self.assertEqual(self.ftl_length_1.get_cost(), 14)
        self.assertEqual(self.ftl_length_2.get_cost(), 11)

    def testFromParent(self):
        self.ftl_length_1.calculate_cost_from_parent(self.ftl_length_0)
        self.assertEqual(self.ftl_length_1.get_cost(), 14)  # parent generation is 0
        self.assertEqual(self.ftl_length_1.generation, 1)

        self.ftl_length_2.calculate_cost_from_parent(self.ftl_length_1)
        self.assertEqual(self.ftl_length_2.get_cost(), 10)
        self.assertEqual(self.ftl_length_2.generation, 2)


if __name__ == '__main__':
    unittest.main()
