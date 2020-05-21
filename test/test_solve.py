import unittest
import pathlib
import pyvista as pv
import numpy as np
from math import pi

from ctrdapp.solve.dynamic_tree import DynamicTree
from ctrdapp.solve.dynamic_tree import Node
from ctrdapp.solve.step import step, step_rotation, get_single_tube_value
from ctrdapp.solve.visualize_utils import parse_json
from ctrdapp.heuristic.only_goal_distance import OnlyGoalDistance
from ctrdapp.heuristic.square_obstacle_avg_plus_weighted_goal import SquareObstacleAvgPlusWeightedGoal


class TestDynamicTree(unittest.TestCase):
    def setUp(self) -> None:
        self.simple_heuristic = OnlyGoalDistance(10)
        heuristic_one = OnlyGoalDistance(1)
        heuristic_two = OnlyGoalDistance(2)
        heuristic_three = OnlyGoalDistance(3)

        self.simple_indices = [0, 0]
        self.other_indices = [1, 1]
        self.simple_g = [[np.eye(4)]]
        self.other_g = [[np.eye(4)]]
        self.other_g[0][0][0:2, 3] = [0.5, 0.1]

        self.zeros = [0.0, 0.0]
        self.ones = [1.0, 1.0]
        self.twos = [2.0, 2.0]
        self.threes = [3.0, 3.0]
        self.tree = DynamicTree(2, self.zeros, self.zeros, self.simple_heuristic, self.simple_g, 10, [1, 6])

        self.final_tree = DynamicTree(2, self.zeros, self.zeros, self.simple_heuristic, self.simple_g, 10, [1, 6])
        self.final_tree.insert(self.ones, self.ones, 0, heuristic_one, self.simple_g, self.simple_indices)
        self.final_tree.insert(self.twos, self.twos, 1, heuristic_two, self.simple_g, self.simple_indices)
        self.final_tree.insert(self.threes, self.threes, 0, heuristic_three, self.other_g, self.other_indices)

    def test_insert(self):
        # self.assertEqual(len(self.tree.kdtree), 1)
        # self.assertEqual(self.tree.map2nodes[0], 0)
        # self.assertEqual(self.tree.kdtree[0].data.all(), self.tree0.data.all())

        self.tree.insert(self.ones, self.ones, 0, self.simple_heuristic, self.simple_g, self.simple_indices)

        # self.assertEqual(len(self.tree.kdtree), 2)
        self.assertEqual(len(self.tree.nodes), 2)
        # self.assertEqual(self.tree.kdtree[1].data.all(), self.tree1.data.all())
        # self.assertEqual(self.tree.kdtree[0], None)
        self.assertEqual(self.tree.nodes[0].children, [1])
        # self.assertEqual(self.tree.map2nodes, [None, 0, 1])

        self.tree.insert(self.twos, self.twos, 0, self.simple_heuristic, self.simple_g, self.simple_indices)

        # self.assertEqual(len(self.tree.kdtree), 2)
        self.assertEqual(len(self.tree.nodes), 3)
        # self.assertEqual(self.tree.kdtree[1].data.all(), self.tree1.data.all())
        # self.assertEqual(self.tree.kdtree[0].data.all(), self.tree2alone.data.all())
        self.assertEqual(self.tree.nodes[0].children, [1, 2])
        # self.assertEqual(self.tree.map2nodes, [2, 0, 1])

        self.tree.insert(self.threes, self.threes, 2, self.simple_heuristic, self.simple_g, self.simple_indices)

        # self.assertEqual(len(self.tree.kdtree), 3)
        self.assertEqual(len(self.tree.nodes), 4)
        # self.assertEqual(self.tree.kdtree[2].data.all(), self.tree3.data.all())
        # self.assertEqual(self.tree.kdtree[1], None)
        # self.assertEqual(self.tree.kdtree[0], None)
        self.assertEqual(self.tree.nodes[2].children, [3])
        # self.assertEqual(self.tree.map2nodes, [None, None, None, 2, 0, 1, 3])

    def test_nearest_neighbor(self):
        self.assertEqual(self.tree.nearest_neighbor([0.0, 0.0]), ([0.0, 0.0], [0.0, 0.0], 0, [0.0, 0.0]))
        self.assertEqual(self.tree.nearest_neighbor([10608.0, -6198745.0]), ([0.0, 0.0], [0.0, 0.0], 0, [0.0, 0.0]))

        self.assertEqual(self.final_tree.nearest_neighbor([0.0, 0.0, 0.0, 0.0]), (self.zeros, self.zeros, 0, self.zeros))
        self.assertEqual(self.final_tree.nearest_neighbor([1.0, 1.0, 1.0, 1.0]), (self.ones, self.ones, 1, self.zeros))
        self.assertEqual(self.final_tree.nearest_neighbor([0.8, 0.8, 0.7, 0.7]), (self.ones, self.ones, 1, self.zeros))
        self.assertEqual(self.final_tree.nearest_neighbor([2.5, 2.5, 2.49, 2.49]), (self.twos, self.twos, 2, self.ones))
        self.assertEqual(self.final_tree.nearest_neighbor([2.5, 2.5, 2.51, 2.51]), (self.threes, self.threes, 3, self.zeros))

    def test_find_all_nearest_neighbor(self):  # todo
        self.assertEqual(self.tree.find_all_nearest_neighbor([0.0, 0.0, 0.0, 0.0], 10), [0])
        self.assertEqual(self.tree.find_all_nearest_neighbor([0.0, 0.0, 0.0, 0.0], 0.0001), [0])
        self.assertEqual(self.tree.find_all_nearest_neighbor([5.0, 5.0, 5.0, 5.0], 1), [])

        self.assertEqual(self.final_tree.find_all_nearest_neighbor([2.0, 1.0001, 1.0, 1.0], 1), [])
        self.assertEqual(self.final_tree.find_all_nearest_neighbor([2.0, 1.0, 1.0, 1.0], 1.0001), [1])
        self.assertEqual(self.final_tree.find_all_nearest_neighbor([2.0, 2.0, 1.0, 1.0], 2.5), [1, 2, 0, 3])
        self.assertEqual(self.final_tree.find_all_nearest_neighbor([2.0, 2.0, 0.0, 0.0], 2.5), [1, 2, 0])

    def test_get_costs(self):
        self.assertEqual(self.final_tree.get_costs(0), [10])
        self.assertEqual(self.final_tree.get_costs(1), [1, 10])
        self.assertEqual(self.final_tree.get_costs(2), [2, 1, 10])
        self.assertEqual(self.final_tree.get_costs(3), [3, 10])

    def test_get_data(self):

        self.assertEqual(self.final_tree.get_tube_data(0), ([self.zeros], [self.zeros], [self.simple_indices]))
        self.assertEqual(self.final_tree.get_tube_data(1), ([self.ones, self.zeros],
                                                            [self.ones, self.zeros],
                                                            [self.simple_indices, self.simple_indices]))
        self.assertEqual(self.final_tree.get_tube_data(2),
                         ([self.twos, self.ones, self.zeros],
                          [self.twos, self.ones, self.zeros],
                          [self.simple_indices, self.simple_indices, self.simple_indices]))
        self.assertEqual(self.final_tree.get_tube_data(3), ([self.threes, self.zeros],
                                                            [self.threes, self.zeros],
                                                            [self.other_indices, self.simple_indices]))

    def test_get_curves(self):
        self.assertEqual(self.final_tree.get_tube_curves(0), [self.simple_g])
        self.assertEqual(self.final_tree.get_tube_curves(2),
                         [self.simple_g, self.simple_g, self.simple_g])
        self.assertEqual(self.final_tree.get_tube_curves(3),
                         [self.other_g, self.simple_g])

    def test_node(self):
        self.assertRaises(ValueError, Node, [0], [0], 2, self.simple_heuristic, self.simple_g, self.simple_indices)

    def test_no_cycle(self):
        simple_heuristic = OnlyGoalDistance(10)
        simple_g = [[np.eye(4)]]
        zeros = [0.0]
        indices = [1]
        tree = DynamicTree(1, zeros, zeros, simple_heuristic, simple_g, 10, [1, 6])  # ind: 0
        tree.insert(zeros, zeros, 0, simple_heuristic, simple_g, indices)  # ind: 1
        tree.insert(zeros, zeros, 1, simple_heuristic, simple_g, indices)  # ind: 2
        tree.insert(zeros, zeros, 2, simple_heuristic, simple_g, indices)  # ind: 3
        tree.insert(zeros, zeros, 0, simple_heuristic, simple_g, indices)  # ind: 4
        tree.insert(zeros, zeros, 4, simple_heuristic, simple_g, indices)  # ind: 5
        tree.insert(zeros, zeros, 4, simple_heuristic, simple_g, indices)  # ind: 6
        tree.insert(zeros, zeros, 2, simple_heuristic, simple_g, indices)  # ind: 7
        tree.insert(zeros, zeros, 7, simple_heuristic, simple_g, indices)  # ind: 8

        """
                              0
                            /   \
                           1     4
                          /     / \
                         2     5   6
                        / \
                       3   7 - 8

        """

        self.assertTrue(tree.no_cycle(5, 3))
        self.assertTrue(tree.no_cycle(5, 2))
        self.assertTrue(tree.no_cycle(5, 1))
        self.assertTrue(tree.no_cycle(5, 7))
        self.assertTrue(tree.no_cycle(5, 6))
        self.assertTrue(tree.no_cycle(6, 5))
        self.assertTrue(tree.no_cycle(4, 1))
        self.assertTrue(tree.no_cycle(8, 3))
        self.assertTrue(tree.no_cycle(4, 6))
        self.assertTrue(tree.no_cycle(2, 3))

        self.assertFalse(tree.no_cycle(8, 1))
        self.assertFalse(tree.no_cycle(8, 2))
        self.assertFalse(tree.no_cycle(7, 1))
        self.assertFalse(tree.no_cycle(3, 1))
        self.assertFalse(tree.no_cycle(8, 7))

        self.assertRaises(ValueError, tree.no_cycle, 0, 5)

    # tests reset_heuristic_all_children as well
    def test_swap_parents(self):
        heuristic_zero = SquareObstacleAvgPlusWeightedGoal(1, 1, 0)
        heuristic_one = SquareObstacleAvgPlusWeightedGoal(1, 1, 0)
        heuristic_two = SquareObstacleAvgPlusWeightedGoal(1, 1, 0)
        heuristic_three = SquareObstacleAvgPlusWeightedGoal(1, 1, 0)
        heuristic_four = SquareObstacleAvgPlusWeightedGoal(1, 1/3, 0)
        heuristic_five = SquareObstacleAvgPlusWeightedGoal(1, 1/2, 0)

        simple_g = [[np.eye(4)]]
        zeros = [0.0]
        indices = [1]
        tree = DynamicTree(1, zeros, zeros, heuristic_zero, simple_g, 1000, [1, 6])  # ind: 0
        tree.insert(zeros, zeros, 0, heuristic_one, simple_g, indices)  # ind: 1
        tree.insert(zeros, zeros, 1, heuristic_two, simple_g, indices)  # ind: 2
        tree.insert(zeros, zeros, 2, heuristic_three, simple_g, indices)  # ind: 3
        tree.insert(zeros, zeros, 2, heuristic_four, simple_g, indices)  # ind: 4
        tree.insert(zeros, zeros, 0, heuristic_five, simple_g, indices)  # ind: 5

        self.assertEqual(heuristic_one.get_cost(), 1)
        self.assertEqual(heuristic_two.get_cost(), 1)
        self.assertEqual(heuristic_three.get_cost(), 1)
        self.assertEqual(heuristic_four.get_cost(), 11/3)
        self.assertEqual(heuristic_five.get_cost(), 4)

        tree.swap_parents(2, 5, heuristic_two, heuristic_five)

        self.assertEqual(heuristic_one.get_cost(), 1)
        self.assertEqual(heuristic_two.get_cost(), 5/2)
        self.assertEqual(heuristic_three.get_cost(), 2)
        self.assertEqual(heuristic_four.get_cost(), 14/3)
        self.assertEqual(heuristic_five.get_cost(), 4)

    def test_calc_new_random_config(self):
        pass


class TestStep(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = [1, 2, 3]
        self.first_tube_neighbor = [5, 2, 3]
        self.second_tube_neighbor = [1, 5, 3]
        self.third_tube_neighbor = [1, 2, 5]
        self.insert_rand = [11, 12, 13]

    def test_step(self):
        self.assertEqual(step([0, 0], [1, 20, 1, 40], 5), [1, 1])
        self.assertEqual(step([1, 0], [5, 20, 0, 40], 3), [4, 0])
        self.assertEqual(step([10, 5], [4, 20, -3, 20], 5), [7, 1])
        self.assertEqual(step([1, 1, 1], [4, 20, 1, 20, 1, 20], 1.5), [2.5, 1, 1])
        self.assertEqual(step([1, 1, 1], [11, 20, 11, 20, 6, 20], 3), [3, 3, 2])
        self.assertRaises(ValueError, step, [0, 20, 0, 20], [1, 1, 1], 0.5)

    def test_step_rotation_single(self):
        new_vec, delta = step_rotation([0], [20, 0.5], 1)
        self.assertEqual(new_vec, [0.5])
        self.assertEqual(delta, [0.5])

        new_vec, delta = step_rotation([0.5], [20, 0], 1)
        self.assertEqual(new_vec, [0])
        self.assertEqual(delta, [-0.5])

        new_vec, delta = step_rotation([6], [20, 0.5], 1)
        self.assertEqual(new_vec, [0.5])
        self.assertEqual(delta, [2*pi - 5.5])

        new_vec, delta = step_rotation([0.5], [20, 6], 1)
        self.assertEqual(new_vec, [6])
        self.assertEqual(delta, [5.5 - 2*pi])

        new_vec, delta = step_rotation([0], [20, 1.5], 1)
        self.assertEqual(new_vec, [1])
        self.assertEqual(delta, [1])

        new_vec, delta = step_rotation([1.5], [20, 0], 1)
        self.assertEqual(new_vec, [0.5])
        self.assertEqual(delta, [-1])

    def test_step_rotation_multiple(self):
        new_vec, delta = step_rotation([0, 0], [20, 1, 20, 1], 2)
        self.assertEqual(new_vec, [1, 1])
        self.assertEqual(delta, [1, 1])

        new_vec, delta = step_rotation([0, 0], [20, 0.3, 20, 0.4], 0.5)
        self.assertEqual(new_vec, [0.3, 0.4])
        self.assertEqual(delta, [0.3, 0.4])

        new_vec, delta = step_rotation([0, 0], [20, 0.6, 20, 0.8], 0.5)
        self.assertEqual(new_vec, [0.3, 0.4])
        self.assertEqual(delta, [0.3, 0.4])

        new_vec, delta = step_rotation([0, 0.4], [20, 0.3, 20, 0], 0.5)
        self.assertEqual(new_vec, [0.3, 0])
        self.assertEqual(delta, [0.3, -0.4])

        new_vec, delta = step_rotation([2*pi - 0.3, 2*pi - 0.4], [20, 0.3, 20, 0.4], 0.5)
        self.assertEqual(new_vec, [0, 0])
        self.assertAlmostEqual(delta[0], 0.3)
        self.assertAlmostEqual(delta[1], 0.4)

    def test_get_delta_rotation(self):
        self.assertTrue(False)

    def test_single_tube_value(self):
        # previous tube selection maintained w/ random < probability
        new_ins, num = get_single_tube_value(
            self.insert_rand, self.first_tube_neighbor, self.parent, 0.8, 0.7999)
        self.assertEqual(new_ins, [11, 2, 3])
        self.assertEqual(num, 0)

        new_ins, num = get_single_tube_value(
            self.insert_rand, self.second_tube_neighbor, self.parent, 0.8, 0)
        self.assertEqual(new_ins, [1, 12, 3])
        self.assertEqual(num, 1)

        new_ins, num = get_single_tube_value(
            self.insert_rand, self.third_tube_neighbor, self.parent, 0.5, 0.4)
        self.assertEqual(new_ins, [1, 2, 13])
        self.assertEqual(num, 2)

        # check boundary cases for other probabilities
        new_ins, num = get_single_tube_value(
            self.insert_rand, self.first_tube_neighbor, self.parent, 0.8, 0.8)
        self.assertEqual(new_ins, [5, 12, 3])
        self.assertEqual(num, 1)

        new_ins, num = get_single_tube_value(
            self.insert_rand, self.first_tube_neighbor, self.parent, 0.8, 0.9)
        self.assertEqual(new_ins, [5, 2, 13])
        self.assertEqual(num, 2)

        new_ins, num = get_single_tube_value(
            self.insert_rand, self.second_tube_neighbor, self.parent, 0.5, 0.749)
        self.assertEqual(new_ins, [11, 5, 3])
        self.assertEqual(num, 0)

        new_ins, num = get_single_tube_value(
            self.insert_rand, self.second_tube_neighbor, self.parent, 0.5, 0.75)
        self.assertEqual(new_ins, [1, 5, 13])
        self.assertEqual(num, 2)


class VisualizeUtils(unittest.TestCase):
    def test_parse_json_box(self):
        path = pathlib.Path().absolute()
        file = path / "configuration" / "init_mesh.json"
        goal = parse_json("goal", file)

        goal_box = pv.Box([19.5, 20.5, -1, 1, -1.5, 1.5])
        self.assertEqual(goal_box.bounds, goal[0].bounds)

    def test_parse_json_not_imp(self):
        path = pathlib.Path().absolute()
        file = path / "configuration" / "init_not_imp.json"

        self.assertRaises(NotImplementedError, parse_json, "obstacles", file)

    @staticmethod
    def test_visualize_mesh():
        path = pathlib.Path().absolute()
        file = path / "configuration" / "init_mesh.json"
        objects = parse_json("obstacles", file)
        goal = parse_json("goal", file)

        plotter = pv.Plotter()
        plotter.add_mesh(objects[0], opacity=0.5)
        plotter.add_mesh(goal[0])
        plotter.add_mesh(pv.Sphere(radius=1), color='r')
        plotter.show()


if __name__ == '__main__':
    unittest.main()
