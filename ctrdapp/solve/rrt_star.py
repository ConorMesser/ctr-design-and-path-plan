from .rrt import RRT
from numpy.linalg import norm
from math import pi


class RRTStar(RRT):

    def __init__(self, model, heuristic_factory, collision_detector, configuration):
        super().__init__(model, heuristic_factory, collision_detector, configuration)
        self.single_tube_control = False  # impossible to maintain with the rewire operation

    def _single_iteration(self):
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
                self.tree.solution.append(new_index)
                self.found_solution = True
                goal_dist = 0
            insert_fractions = [1 - (float(i) / self.insert_max) for i in true_insertion]  # inverse insertion
            insert_frac = sum(insert_fractions)  # = 0 if all tubes are fully inserted; = tube_num if fully retracted
            new_node_heuristic = self.heuristic_factory.create(min_obstacle_distance=obs_min,
                                                               goal_distance=goal_dist,
                                                               follow_the_leader=ftl_heuristic,
                                                               insertion_fraction=insert_frac)

            neighbor_indices = self.tree.find_all_nearest_neighbor(true_insertion, self.step_bound * 1.5)
            neighbor_indices = [index for index in neighbor_indices if
                                norm_rotation_difference(self.tree.nodes[index].rotation, new_rotation) <= self.rotation_max]
            neighbor_costs = [new_node_heuristic.test_cost_from_parent(self.tree.nodes[index].heuristic)
                              for index in neighbor_indices]

            index_min = min(range(len(neighbor_costs)), key=neighbor_costs.__getitem__)
            best_neighbor_as_parent = neighbor_indices[index_min]
            neighbor_indices.remove(best_neighbor_as_parent)

            self.tree.insert(true_insertion, new_rotation, best_neighbor_as_parent,
                             new_node_heuristic, this_g, insert_indices)  # todo implement lazy insert/collision check

            for ind in neighbor_indices:
                neighbor_heuristic = self.tree.nodes[ind].heuristic
                this_cost = neighbor_heuristic.get_cost()
                new_cost = neighbor_heuristic.test_cost_from_parent(new_node_heuristic)
                if new_cost < this_cost and self.tree.no_cycle(new_index, ind):
                    self.tree.swap_parents(ind, new_index, neighbor_heuristic, new_node_heuristic)


def norm_rotation_difference(rotation_one, rotation_two):
    diff = [r2 - r1 for r1, r2 in zip(rotation_one, rotation_two)]
    new_diff = []
    for d in diff:
        pos_d = abs(d)
        if pos_d > pi:
            new_diff.append(2*pi - pos_d)
        else:
            new_diff.append(pos_d)
    return norm(new_diff)
