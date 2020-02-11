from .rrt import RRT
from random import random
from .dynamic_tree import DynamicTree


class RRTStar(RRT):

    def __init__(self, model, heuristic_factory, collision_detector, configuration):

        self.rewire_prob = configuration.get("rewire_probability")
        super().__init__(model, heuristic_factory, collision_detector, configuration)

    def _solve(self):
        if self.found_solution and random() < self.rewire_prob:
            # optimize current solution
            pass  # todo
        else:
            self._single_iteration()
