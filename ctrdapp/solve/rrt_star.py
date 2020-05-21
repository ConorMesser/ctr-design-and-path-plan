from .rrt import RRT
from .step import get_delta_rotation
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
        # todo ftl_heuristic isn't really correct if I change the parent (changes the deltas...)
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
            query_pt = true_insertion + new_rotation
            query_pt[::2] = true_insertion
            query_pt[1::2] = new_rotation
            neighbor_indices = self.tree.find_all_nearest_neighbor(query_pt, self.step_bound)  # think more about limit todo
            # need to calculate ftl_heuristic from each neighbor based on the delta_rotation/delta_insert for coming
            # from each different neighbor. Could use a smaller method in model to do so without calculating this_g
            # this is only necessary if the ftl_heuristic is being used (or some history-/path-dependent heuristic)
            # test cost from parent may need to be altered

            if True:
                cost_arr = []
                heur_arr = []
                for neighbor_ind in neighbor_indices:  # how to skip first neighbor/parent? todo
                    ins_neighbor = self.tree.nodes[neighbor_ind].insertion
                    rot_neighbor = self.tree.nodes[neighbor_ind].rotation
                    g_neighbor = self.tree.nodes[neighbor_ind].g_curves
                    delta_ins_neighbor = [t - p for t, p in zip(true_insertion, ins_neighbor)]
                    delta_rot_neighbor = [get_delta_rotation(p, t) for p, t in zip(rot_neighbor, new_rotation)]
                    _, _, _, _, ftl_neighbor = self.model.solve_integrate(delta_rot_neighbor, delta_ins_neighbor,
                                                                          new_rotation, true_insertion, g_neighbor)
                    # todo don't calculate g_out unnecessarily
                    new_neighbor_heuristic = self.heuristic_factory.create(min_obstacle_distance=obs_min,
                                                                           goal_distance=goal_dist,
                                                                           follow_the_leader=ftl_neighbor,
                                                                           insertion_fraction=insert_frac)
                    this_neighbor_cost = new_neighbor_heuristic.test_cost_from_parent(self.tree.nodes[neighbor_ind].heuristic)
                    cost_arr.append(this_neighbor_cost)
                    heur_arr.append(new_neighbor_heuristic)


            # neighbor_costs = [new_node_heuristic.test_cost_from_parent(self.tree.nodes[index].heuristic)
            #                       for index in neighbor_indices]

            index_min = min(range(len(cost_arr)), key=cost_arr.__getitem__)
            best_neighbor_as_parent = neighbor_indices[index_min]
            parent_heuristic = heur_arr[index_min]

            self.tree.insert(true_insertion, new_rotation, best_neighbor_as_parent,
                             parent_heuristic, this_g, insert_indices)  # todo implement lazy insert/collision check

            neighbor_indices.remove(best_neighbor_as_parent)
            for ind in neighbor_indices:
                neighbor_heuristic = self.tree.nodes[ind].heuristic
                this_cost = neighbor_heuristic.get_cost()
                # new_cost = neighbor_heuristic.test_cost_from_parent(parent_heuristic)
                try:
                    no_cycle = self.tree.no_cycle(new_index, ind)
                except RecursionError:
                    print(f'Recursion not working on parent: {new_index} and child: {ind}')
                else:
                    if no_cycle:
                        this_ins = self.tree.nodes[ind].insertion
                        this_rot = self.tree.nodes[ind].rotation
                        g_neighbor = this_g
                        delta_ins_neighbor = [t - p for t, p in zip(this_ins, true_insertion)]
                        delta_rot_neighbor = [get_delta_rotation(p, t) for t, p in zip(this_rot, new_rotation)]

                        _, _, _, _, ftl_neighbor = self.model.solve_integrate(delta_rot_neighbor, delta_ins_neighbor,
                                                                              this_rot, this_ins, g_neighbor)

                        # todo don't calculate g_out unnecessarily
                        new_neighbor_heuristic = self.heuristic_factory.create(min_obstacle_distance=obs_min,
                                                                               goal_distance=goal_dist,
                                                                               follow_the_leader=ftl_neighbor,
                                                                               insertion_fraction=insert_frac)
                        this_neighbor_cost = new_neighbor_heuristic.test_cost_from_parent(parent_heuristic)
                        if this_neighbor_cost < this_cost:
                            self.tree.swap_parents(ind, new_index, neighbor_heuristic, parent_heuristic)



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
