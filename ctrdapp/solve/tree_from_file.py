from .rrt import RRT
from ctrdapp.heuristic.heuristic_factory import create_heuristic_factory

import pathlib
from collections import deque
from distutils.util import strtobool


class TreeFromFile(RRT):

    def __init__(self, model, heuristic_factory, collision_detector, configuration):
        path = pathlib.Path().absolute()
        filename = path / 'output' / configuration.get('run_identifier') / 'tree.txt'

        # must order nodes from file with respect to parent
        self.node_order, self.nodes_list = order_nodes(filename)

        configuration['heuristic_type'] = "only_goal_distance"
        heuristic_factory = create_heuristic_factory(configuration, {'only_goal_distance': []})
        self.delta_x = configuration.get('delta_x')
        self.tube_lengths = configuration.get('tube_lengths')

        super().__init__(model, heuristic_factory, collision_detector, configuration)

    def _single_iteration(self):
        this_ind = self.node_order.popleft()
        this_node = self.nodes_list[this_ind]

        if this_node.get('goal'):
            self.tree.at_goal.append(len(self.tree.nodes))  # before insertion, so no -1
            self.found_solution = True

        # get insertion indices from the values (inverse insertion)
        insertion_indices = [int((l - ins) / self.delta_x) for ins, l in zip(this_node.get('ins'), self.tube_lengths)]
        this_g = self.model.solve_g(insertion_indices, this_node.get('rot'), full=False)
        new_heuristic = self.heuristic_factory.create(goal_distance=this_node.get('cost'))
        self.tree.insert(this_node.get('ins'), this_node.get('rot'), this_node.get('parent'),
                         new_heuristic, this_g, insertion_indices)


def order_nodes(file):
    parent_list = {}
    node_list = []
    node_order = deque()
    with open(file, 'r') as tree:
        for this_node in tree:
            this_node = this_node.split(' | ')
            this_ind = int(this_node[0])
            insertion = this_node[1].split(', ')
            insertion = [int(ins) for ins in insertion]
            rotation = this_node[2].split(', ')
            rotation = [float(rot) for rot in rotation]

            cost = float(this_node[4])
            goal_found = bool(strtobool(this_node[5].rstrip()))
            if this_ind == 0:
                parent = None
            else:
                parent = int(this_node[3])

            node_list.append({'ins': insertion, 'rot': rotation, 'parent': parent, 'cost': cost, 'goal': goal_found})
            curr_list = parent_list.get(parent)
            if curr_list is None:
                parent_list[parent] = [this_ind]
            else:
                curr_list.append(this_ind)

    queue = deque(parent_list.get(0))
    node_order.extend(parent_list.get(0))
    while queue:
        ind = queue.popleft()
        new_indices = parent_list.get(ind)
        if new_indices:
            node_order.extend(new_indices)
            queue.extend(new_indices)
            for i in new_indices:
                this_node = node_list[i]
                this_node['parent'] = node_order.index(ind) + 1

    return node_order, node_list
