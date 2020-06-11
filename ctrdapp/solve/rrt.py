from math import pi, sqrt
import numpy as np

from .solver import Solver
from random import random
from .dynamic_tree import DynamicTree
from .step import step, step_rotation, get_single_tube_value
from ..heuristic.heuristic_factory import HeuristicFactory
from .visualize_utils import visualize_curve, visualize_curve_single, visualize_tree


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
        self.rotation_max = 0.1745  # 0.1745  # 10 degrees in either direction todo make this variable based on insertion?
        """float : maximum rotation from nearest neighbor, in Radians"""
        self.insert_max = configuration.get("insertion_max")
        """float : maximum tube/insertion length"""
        self.found_solution = False  # todo is this necessary?
        """boolean : has any Node collided with the goal?"""
        self.single_tube_control = configuration.get("single_tube_control")
        """boolean : can only one tube be controlled (inserted/rotated) per step?"""

        # initial heuristic is not used (generation is set as 0)
        ftl = []
        for i in range(self.tube_num):
            ftl.append([np.asarray([0, 0, 0, 100, 0, 0])])
        init_heuristic = self.heuristic_factory.create(min_obstacle_distance=0.00001,  # todo
                                                       goal_distance=10e10,
                                                       follow_the_leader=ftl,
                                                       insertion_fraction=self.tube_num)
        init_tube = [np.eye(4)]
        init_g_curves = [init_tube] * self.tube_num
        scaling = [1, 1] # [1, self.step_bound / self.rotation_max]

        self.tree = DynamicTree(self.tube_num, [0] * self.tube_num, [0] * self.tube_num, init_heuristic, init_g_curves,
                                self.iter_max, scaling)
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

        ***The nearest neighbor is only found by insertion values.
        Accordingly, the rotation values are just incremented by random values
        bounded by the rotation_max. This is a static value for now but may be
        extended to decrease as insertion length increases (so the actual change
        in tip distance will remain constant.***

        Returns
        -------
        VOID
        """

        delta_rotation, delta_insert, new_rotation, new_insert, neighbor_index, g_neighbor = self.calc_new_random_config()
        this_g, this_eta, insert_indices, true_insertion, ftl_heuristic = self.model.solve_integrate(delta_rotation,
                                                                                                     delta_insert,
                                                                                                     new_rotation,
                                                                                                     new_insert,
                                                                                                     g_neighbor)
        obs_min, goal_dist = self.cd.check_collision(this_g, self.tube_rad)
        if obs_min < 0:
            pass
        else:
            new_index = len(self.tree.nodes)  # before insertion, so no -1
            if goal_dist < 0:
                self.tree.at_goal.append(new_index)
                self.found_solution = True
                goal_dist = 0
            insert_fractions = [1 - (float(i) / self.insert_max) for i in true_insertion]  # inverse insertion
            insert_frac = sum(insert_fractions)  # = 0 if all tubes are fully inserted; = tube_num if fully retracted
            new_heuristic = self.heuristic_factory.create(min_obstacle_distance=obs_min,
                                                          goal_distance=goal_dist,
                                                          follow_the_leader=ftl_heuristic,
                                                          insertion_fraction=insert_frac)
            self.tree.insert(true_insertion, new_rotation, neighbor_index,
                             new_heuristic, this_g, insert_indices)  # todo implement lazy insert/collision check

    def calc_new_random_config(self):
        # choose random config limited by the max insertion distance
        tube_control_params = []
        for _ in range(self.tube_num):
            tube_control_params.append(random() * self.insert_max)
            tube_control_params.append(random() * 2 * pi)

        insert_neighbor, rotation_neighbor, neighbor_index, neighbor_parent = self.tree.nearest_neighbor(tube_control_params)
        if self.single_tube_control:
            tube_control_params, controlled_tube_num = get_single_tube_value(
                tube_control_params, insert_neighbor, rotation_neighbor, neighbor_parent, 0.8, random())
        new_insert = step(insert_neighbor, tube_control_params, sqrt(self.step_bound))
        new_rotation, delta_rotation = step_rotation(rotation_neighbor, tube_control_params, sqrt(self.step_bound)*self.rotation_max/self.step_bound)
        delta_insert = [new - old for new, old in zip(new_insert, insert_neighbor)]
        g_neighbor = self.tree.nodes[neighbor_index].g_curves

        return delta_rotation, delta_insert, new_rotation, new_insert, neighbor_index, g_neighbor

    def get_path(self, index):
        g_out = self.tree.get_tube_curves(index)
        insert, rotate, insert_indices = self.tree.get_tube_data(index)
        return g_out, insert, rotate, insert_indices

    def get_best_cost(self):
        if self.found_solution:
            solution_ind_list = self.tree.at_goal
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

    def save_best_solution(self, output_dir):
        """

        Parameters
        ----------
        output_dir : pathlib.PosixPath

        Returns
        -------

        """
        filename = output_dir / "solution_path.txt"
        solution_file = open(filename, "w")
        # with filename.open('w') as solution_file:

        cost, best_index = self.get_best_cost()
        solution_file.write(f"Cost: {cost}\n")
        solution_list = self.tree.get_index_list(best_index)
        solution_list.reverse()

        ins_str = ", ".join(str(ind) for ind in solution_list)
        solution_file.write(ins_str)

        solution_file.close()

    def save_tree(self, output_dir):
        self.tree.save_tree(output_dir)

    def visualize_best_solution(self, objects_file, output_dir):
        _, best_index = self.get_best_cost()
        self.visualize_from_index(best_index, objects_file, output_dir, "best_solution")

    def visualize_best_solution_path(self, objects_file, output_dir):
        _, best_index = self.get_best_cost()
        self.visualize_from_index_path(best_index, objects_file, output_dir, "best_solution_animated")

    def visualize_from_index(self, index, objects_file, output_dir, filename):
        g_out, insert, rotate, insert_indices = self.get_path(index)
        g_out_flat = g_out[0]
        visualize_curve_single(g_out_flat, objects_file, self.tube_num, self.tube_rad, output_dir, filename)

    def visualize_from_index_path(self, index, objects_file, output_dir, filename):
        g_out, insert, rotate, insert_indices = self.get_path(index)
        g_out_flat_list = []
        for curve in reversed(g_out):
            g_out_flat_list.append(curve)
        visualize_curve(g_out_flat_list, objects_file, self.tube_num, self.tube_rad, output_dir, filename)

    def visualize_full_search(self, output_dir, tube_num=0, with_solution=True):

        from_list = []
        to_list = []
        node_list = []
        cost_list = []
        # start at root of tree
        root_node = self.tree.nodes[0]
        root_data = [root_node.insertion[tube_num], root_node.rotation[tube_num]]
        # collect from and to information, recur over children
        for i in root_node.children:
            child_from_l, child_to_l, children_nodes, children_costs = self._tree_search_recur(
                i, root_data, tube_num)
            from_list = from_list + child_from_l
            to_list = to_list + child_to_l
            node_list = node_list + children_nodes
            cost_list = cost_list + children_costs

        if with_solution:
            _, best_index = self.get_best_cost()
            solution_list = self.tree.get_index_list(best_index)
        else:
            solution_list = []

        visualize_tree(from_list, to_list, node_list, output_dir, "full_tree",
                       solution_list, self.tree.at_goal, cost_list)

    def _tree_search_recur(self, child_index, parent_data, tube_num):

        this_child = self.tree.nodes[child_index]
        from_list = [parent_data]
        child_data = [this_child.insertion[tube_num], this_child.rotation[tube_num]]
        to_list = [child_data]
        node_list = [child_index]
        cost_list = [this_child.heuristic.get_own_cost()]  # [this_child.heuristic.get_cost()]

        if not this_child.children:
            return from_list, to_list, node_list, cost_list
        else:
            for ind in this_child.children:
                children_from, children_to, children_nodes, children_costs = self._tree_search_recur(
                    ind, child_data, tube_num)
                from_list = from_list + children_from
                to_list = to_list + children_to
                node_list = node_list + children_nodes
                cost_list = cost_list + children_costs
            return from_list, to_list, node_list, cost_list
