from math import pi
import numpy as np

from .solver import Solver
from random import random
from .dynamic_tree import DynamicTree
from .step import step, get_single_tube_value
from ..heuristic.heuristic_factory import HeuristicFactory
from .visualize_utils import visualize_curve, visualize_curve_single, visualize_tree
from ..model.model import truncate_g


class RRT(Solver):
    """Rapidly-exploring Random Tree algorithm with a KD-dynamic-tree.

    Implements the common sampling-based planner, RRT, which uses randomly
    generated points to expand the search tree to find the goal, deleting any
    points that are in collision with obstacles.
    """

    def __init__(self, model, heuristic_factory, collision_detector, configuration):
        """
        Parameters
        ----------
        model : Model
            Model object containing design info for tube deformation calculation
        heuristic_factory : HeuristicFactory
            Heuristic object containing general parameters and method to create
            new Heuristic object
        collision_detector : CollisionChecker
            Contains obstacles, goal, and methods for collision queries
        configuration : dict
            Dictionary storing configuration variables
        """
        super().__init__(model, heuristic_factory, collision_detector, configuration)
        """Assigns model, heuristic, collision_detector, tube_num, and tube_rad"""

        self.iter_max = configuration.get("iteration_number")
        """int : number of iterations"""
        self.step_bound = configuration.get("step_bound")
        """float : maximum step size (from nearest neighbor) for new config"""
        self.rotation_max = 0.349  # 10 degrees in either direction todo make this variable based on insertion?
        """float : maximum rotation from nearest neighbor, in Radians"""
        self.insert_max = configuration.get("insertion_max")
        """float : maximum tube/insertion length"""
        self.nn_func = configuration.get("nearest_neighbor_function")
        """int : the distance function used by numpy.spatial.KDTree
        1 = Manhattan, 2 = Euclidean, infinity = maximum coordinate diff"""
        self.found_solution = False  # todo is this necessary?
        """boolean : has any Node collided with the goal?"""
        self.single_tube_control = configuration.get("single_tube_control")
        """boolean : can only one tube be controlled (inserted/rotated) per step?"""

        # calculate initial obstacle/goal distances and initialize the heuristic
        init_tube = [np.eye(4), np.eye(4)]
        init_tube[1][0, 3] = 0.0001
        init_tubes = [init_tube] * self.tube_num
        obstacle_min_dist, goal_dist = self.cd.check_collision(init_tubes, self.tube_rad)
        init_heuristic = self.heuristic_factory.create(obstacle_min_dist, goal_dist)
        dummy_heuristic = self.heuristic_factory.create(obstacle_min_dist, goal_dist)
        init_heuristic.calculate_cost_from_parent(dummy_heuristic)
        init_g_curves = self.model.solve_g()

        self.tree = DynamicTree(self.tube_num, [0] * self.tube_num,
                                [0] * self.tube_num, init_heuristic, init_g_curves)
        """DynamicTree : stores the tree information"""
        self._solve()  # rrt is solved at initialization

    def _solve(self):
        """Implements the rrt algorithm based on the given parameters.

        Most of the implementation is contained in the _single_iteration method
        to allow for abstraction for other rrt child classes. Modifies the tree
        and found_solution fields.

        Returns
        -------
        VOID
        """

        for _ in range(self.iter_max):
            self._single_iteration()

    def _single_iteration(self):
        """Runs one iteration of the rrt algorithm.

        Gives a point between a randomly selected insertion configuration and
        its nearest neighbor. Checks this new configuration for collisions; if
        there is no collision, it is added to the tree and the cost and solution
        fields are updated. Modifies the tree and found_solution.

        ***currently alters the insertion and rotation values for all tubes in
        each step. The nearest neighbor is only found by insertion values.
        Accordingly, the rotation values are just incremented by random values
        bounded by the rotation_max. This is a static value for now but may be
        extended to decrease as insertion length increases (so the actual change
        in tip distance will remain constant.***

        Returns
        -------
        VOID
        """

        # choose random config limited by the max insertion distance
        insert_rand = []
        for _ in range(self.tube_num):
            insert_rand.append(random() * self.insert_max)

        insert_neighbor, neighbor_index, neighbor_parent = self.tree.nearest_neighbor(insert_rand)
        if self.single_tube_control:
            insert_rand, controlled_tube_num = get_single_tube_value(
                insert_rand, insert_neighbor, neighbor_parent, 0.8, random())
        new_insert = step(insert_neighbor, insert_rand, self.step_bound)

        delta_insert = [new - old for new, old in zip(new_insert, insert_neighbor)]

        rotation_neighbor = self.tree.nodes[neighbor_index].rotation
        new_rotation = []
        delta_rotation = []
        for rot in rotation_neighbor:
            this_delta_rotation = (random() - 0.5) * self.rotation_max
            delta_rotation.append(this_delta_rotation)
            this_rot = this_delta_rotation + rot
            new_rotation.append(this_rot % (pi * 2))  # keeps rotations as [0, 2*pi)

        # collision check
        g_previous = self.tree.nodes[neighbor_index].g_curves

        # ------ input could change with space iteration
        #   insert_self instead of insert_neighbor
        #   rotation_self instead of g_previous
        # ----- output
        #   this_g would already be truncated
        this_g, this_eta, insert_indices, true_insertion = self.model.solve_iterate(
            delta_rotation, delta_insert, insert_neighbor, g_previous)
        this_g_truncated = truncate_g(this_g, insert_indices)
        obs_min, goal_dist = self.cd.check_collision(this_g_truncated, self.tube_rad)
        if obs_min < 0:
            pass
        else:
            new_index = len(self.tree.nodes)  # before insertion, so no -1
            if goal_dist < 0:
                self.tree.solution.append(new_index)
                self.found_solution = True
                goal_dist = 0
            new_heuristic = self.heuristic_factory.create(obs_min, goal_dist)
            self.tree.insert(true_insertion, new_rotation, neighbor_index,
                             new_heuristic, this_g, insert_indices)  # todo implement lazy insert/collision check

    def get_path(self, index):
        g_out = self.tree.get_tube_curves(index)
        insert, rotate, insert_indices = self.tree.get_tube_data(index)
        return g_out, insert, rotate, insert_indices

    def get_best_cost(self):
        if self.found_solution:
            solution_ind_list = self.tree.solution
        else:
            solution_ind_list = range(len(self.tree.nodes))
        best_index = 0
        best_cost = 1e200  # a very large number
        for i in solution_ind_list:
            this_cost = self.tree.nodes[i].get_cost()
            if this_cost < best_cost:
                best_cost = this_cost
                best_index = i
        return best_cost, best_index

    def visualize_best_solution(self, objects_file):
        _, best_index = self.get_best_cost()
        self.visualize_from_index(best_index, objects_file)

    def visualize_best_solution_path(self, objects_file):
        _, best_index = self.get_best_cost()
        self.visualize_from_index_path(best_index, objects_file)

    def visualize_from_index(self, index, objects_file):
        g_out, insert, rotate, insert_indices = self.get_path(index)
        # get insert_indices from either insert or from model?
        g_out_truncated = truncate_g(g_out[0], insert_indices[0])
        visualize_curve_single(g_out_truncated, objects_file, self.tube_num, self.tube_rad)

    def visualize_from_index_path(self, index, objects_file):
        g_out, insert, rotate, insert_indices = self.get_path(index)
        g_out_truncated = []
        for g, ind in zip(reversed(g_out), reversed(insert_indices)):
            g_out_truncated.append(truncate_g(g, ind))
        visualize_curve(g_out_truncated, objects_file, self.tube_num, self.tube_rad)

    def visualize_full_search(self):  # todo how to make more useful?
        # check how many dim (greater than 3 unsupported? todo)
        if self.tube_num > 3:
            pass

        from_list = []
        to_list = []
        node_list = []
        # start at root of tree
        root_node = self.tree.nodes[0]
        # collect from and to information, recur over children
        for i in root_node.children:
            child_from_l, child_to_l, children_nodes = self._tree_search_recur(
                i, root_node.insertion)
            from_list = from_list + child_from_l
            to_list = to_list + child_to_l
            node_list = node_list + children_nodes

        visualize_tree(from_list, to_list, node_list)

    def _tree_search_recur(self, child_index, parent_insertion):

        this_child = self.tree.nodes[child_index]
        from_list = [parent_insertion]
        to_list = [this_child.insertion]
        node_list = [child_index]

        if not this_child.children:
            return from_list, to_list, node_list
        else:
            for ind in this_child.children:
                children_from, children_to, children_nodes = self._tree_search_recur(
                    ind, this_child.insertion)
                from_list = from_list + children_from
                to_list = to_list + children_to
                node_list = node_list + children_nodes
            return from_list, to_list, node_list
