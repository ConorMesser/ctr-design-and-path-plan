import unittest
from scipy import spatial
import pathlib
import pyvista as pv
import numpy as np

from ctrdapp.solve.dynamic_tree import DynamicTree
from ctrdapp.solve.dynamic_tree import Node
from ctrdapp.solve.step import step, get_single_tube_value
from ctrdapp.solve.visualize_utils import parse_json
from ctrdapp.heuristic.only_goal_distance import OnlyGoalDistance


class TestDynamicTree(unittest.TestCase):
    def setUp(self) -> None:
        self.simple_heuristic = OnlyGoalDistance(10)
        self.heuristic_one = OnlyGoalDistance(1)
        self.heuristic_two = OnlyGoalDistance(2)
        self.heuristic_three = OnlyGoalDistance(3)
        self.simple_indices = [0, 0]
        self.other_indices = [1, 1]
        self.simple_g = [[np.eye(4)]]
        self.other_g = [[np.eye(4)]]
        self.other_g[0][0][0:2, 3] = [0.5, 0.1]

        self.zeros = [0.0, 0.0]
        self.ones = [1.0, 1.0]
        self.twos = [2.0, 2.0]
        self.threes = [3.0, 3.0]
        self.tree = DynamicTree(2, self.zeros, self.zeros, self.simple_heuristic, self.simple_g)

        self.final_tree = DynamicTree(2, self.zeros, self.zeros, self.simple_heuristic, self.simple_g)
        self.final_tree.insert(self.ones, self.ones, 0, self.simple_heuristic, self.simple_g, self.simple_indices)
        self.final_tree.insert(self.twos, self.twos, 1, self.simple_heuristic, self.simple_g, self.simple_indices)
        self.final_tree.insert(self.threes, self.threes, 0, self.simple_heuristic, self.other_g, self.other_indices)

        self.tree0 = spatial.KDTree([self.zeros])
        self.tree1 = spatial.KDTree([self.zeros, self.ones])
        self.tree2 = spatial.KDTree([self.zeros, self.ones, self.twos])
        self.tree3 = spatial.KDTree([self.zeros, self.ones, self.twos, self.threes])
        self.tree2alone = spatial.KDTree([self.twos])

    # insert tests
    # testing state changes:
    # prev kdtrees empty, new kdtree
    # children set
    # map2nodes updated
    def test_insert(self):
        self.assertEqual(len(self.tree.kdtrees), 1)
        self.assertEqual(self.tree.map2nodes[0], 0)
        self.assertEqual(self.tree.kdtrees[0].data.all(), self.tree0.data.all())

        self.tree.insert(self.ones, self.ones, 0, self.simple_heuristic, self.simple_g, self.simple_indices)

        self.assertEqual(len(self.tree.kdtrees), 2)
        self.assertEqual(len(self.tree.nodes), 2)
        self.assertEqual(self.tree.kdtrees[1].data.all(), self.tree1.data.all())
        self.assertEqual(self.tree.kdtrees[0], None)
        self.assertEqual(self.tree.nodes[0].children, [1])
        self.assertEqual(self.tree.map2nodes, [None, 0, 1])

        self.tree.insert(self.twos, self.twos, 0, self.simple_heuristic, self.simple_g, self.simple_indices)

        self.assertEqual(len(self.tree.kdtrees), 2)
        self.assertEqual(len(self.tree.nodes), 3)
        self.assertEqual(self.tree.kdtrees[1].data.all(), self.tree1.data.all())
        self.assertEqual(self.tree.kdtrees[0].data.all(), self.tree2alone.data.all())
        self.assertEqual(self.tree.nodes[0].children, [1, 2])
        self.assertEqual(self.tree.map2nodes, [2, 0, 1])

        self.tree.insert(self.threes, self.threes, 2, self.simple_heuristic, self.simple_g, self.simple_indices)

        self.assertEqual(len(self.tree.kdtrees), 3)
        self.assertEqual(len(self.tree.nodes), 4)
        self.assertEqual(self.tree.kdtrees[2].data.all(), self.tree3.data.all())
        self.assertEqual(self.tree.kdtrees[1], None)
        self.assertEqual(self.tree.kdtrees[0], None)
        self.assertEqual(self.tree.nodes[2].children, [3])
        self.assertEqual(self.tree.map2nodes, [None, None, None, 2, 0, 1, 3])

    def test_nearest_neighbor(self):
        self.assertEqual(self.tree.nearest_neighbor([0.0, 0.0]), ([0.0, 0.0], 0))
        self.assertEqual(self.tree.nearest_neighbor([10608.0, -6198745.0]), ([0.0, 0.0], 0))

        self.tree.insert(self.ones, self.ones, 0, self.simple_heuristic, self.simple_g, self.simple_indices)
        self.tree.insert(self.twos, self.twos, 0, self.simple_heuristic, self.simple_g, self.simple_indices)
        self.tree.insert(self.threes, self.threes, 0, self.simple_heuristic, self.simple_g, self.simple_indices)

        self.assertEqual(self.tree.nearest_neighbor([0.0, 0.0]), ([0.0, 0.0], 0))
        self.assertEqual(self.tree.nearest_neighbor([1.0, 1.0]), (self.ones, 1))
        self.assertEqual(self.tree.nearest_neighbor([0.8, 0.7]), (self.ones, 1))
        self.assertEqual(self.tree.nearest_neighbor([2.5, 2.49]), (self.twos, 2))
        self.assertEqual(self.tree.nearest_neighbor([2.5, 2.51]), (self.threes, 3))

    def test_get_costs(self):
        self.assertEqual(self.tree.get_costs(0), [10])

        self.tree.insert(self.ones, self.ones, 0, self.heuristic_one, self.simple_g, self.simple_indices)
        self.tree.insert(self.twos, self.twos, 1, self.heuristic_two, self.simple_g, self.simple_indices)
        self.tree.insert(self.threes, self.threes, 0, self.heuristic_three, self.simple_g, self.simple_indices)

        self.assertEqual(self.tree.get_costs(0), [10])
        self.assertEqual(self.tree.get_costs(1), [1, 10])
        self.assertEqual(self.tree.get_costs(2), [2, 1, 10])
        self.assertEqual(self.tree.get_costs(3), [3, 10])

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

    def test_assign(self):
        list1 = [0, 1, 2]
        DynamicTree._assign(list1, 3, 3)
        self.assertEqual(list1, [0, 1, 2, 3])

        list2 = [0, 1, 2]
        DynamicTree._assign(list2, 2, 5)
        self.assertEqual(list2, [0, 1, 5])

        list3 = [0, 1, 2]
        self.assertRaises(UserWarning, DynamicTree._assign, list3, 20, 5)
        self.assertEqual(list3, [0, 1, 2, 5])


class TestStep(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = [1, 2, 3]
        self.first_tube_neighbor = [5, 2, 3]
        self.second_tube_neighbor = [1, 5, 3]
        self.third_tube_neighbor = [1, 2, 5]
        self.insert_rand = [11, 12, 13]

    def test_step(self):
        self.assertEqual(step([0, 0], [1, 1], 5), [1, 1])
        self.assertEqual(step([1, 0], [5, 0], 3), [4, 0])
        self.assertEqual(step([10, 5], [4, -3], 5), [7, 1])
        self.assertEqual(step([1, 1, 1], [4, 1, 1], 1.5), [2.5, 1, 1])
        self.assertEqual(step([1, 1, 1], [11, 11, 6], 3), [3, 3, 2])
        self.assertRaises(ValueError, step, [0, 0], [1, 1, 1], 0.5)

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
