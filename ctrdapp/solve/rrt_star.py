from .rrt import RRT
from random import random
from .dynamic_tree import DynamicTree


class RRTStar(RRT):

    def __init__(self, model, heuristic_factory, collision_detector, configuration):

        self.rewire_prob = configuration.get("rewire_probability")
        super().__init__(model, heuristic_factory, collision_detector, configuration)

    '''
    Get new random node (same procedure) 
    check if x is collision-free; if not, discard and continue
    Get list of nearest neighbors to x***
        Discard any that are too far rotationally
    Pick best parent (neighbor) node based on cost
    Insert x
    Rewire node using same list of nearest neighbors
        Compare curr cost of nearest neighbors to cost if changing parent to x***
        If less than:
            Check for cycle and add edge (change parent, cost, children lists)***
    '''
    def _solve(self):
        if self.found_solution and random() < self.rewire_prob:
            # optimize current solution
            pass  # todo
        else:
            self._single_iteration()
