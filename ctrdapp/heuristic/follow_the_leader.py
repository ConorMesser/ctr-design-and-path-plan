from .heuristic import Heuristic
import numpy as np


class FollowTheLeader(Heuristic):

    def __init__(self, only_tip, follow_the_leader):
        if only_tip:
            self.avg_ftl = follow_the_leader[-1][-1]  # heuristic for the tip
        else:
            self.avg_ftl = np.mean(follow_the_leader)  # time considerations

        self.generation = 0
        self.cost = self._calculate_cost()

    def calculate_cost_from_parent(self, parent: 'FollowTheLeader'):
        if self.generation != 0:
            print(f"Cost already calculated. Do not run method twice.")
            return
        self.avg_ftl = self._calculate_avg(parent.avg_ftl, parent.generation, self.avg_ftl)
        self.generation = parent.generation + 1
        self.cost = self._calculate_cost()

    def _calculate_cost(self):
        return self.avg_ftl

    @staticmethod
    def _calculate_avg(parent_avg, parent_gen, this_ftl):
        prior_sum = parent_gen * parent_avg
        return (prior_sum + this_ftl) / (parent_gen + 1)

    def get_cost(self):
        return self.cost
