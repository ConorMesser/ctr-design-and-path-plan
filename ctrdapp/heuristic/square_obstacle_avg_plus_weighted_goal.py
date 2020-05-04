from .heuristic import Heuristic


class SquareObstacleAvgPlusWeightedGoal(Heuristic):

    def __init__(self, goal_weight, this_obstacle_min, goal_dist):
        self.this_obstacle_min = (1 / this_obstacle_min)**2
        self.avg_obstacle_min = self.this_obstacle_min
        self.generation = 0
        self.goal_weight = goal_weight
        self.goal_dist = goal_dist
        self.cost = self._calculate_cost()

    def calculate_cost_from_parent(self, parent: 'SquareObstacleAvgPlusWeightedGoal', reset=False):
        if self.generation != 0 and not reset:
            print(f"Cost already calculated. Do not run method twice.")
            return
        self.avg_obstacle_min = self._calculate_avg(parent.avg_obstacle_min,
                                                    parent.generation,
                                                    self.this_obstacle_min)
        self.generation = parent.generation + 1
        self.cost = self._calculate_cost()

    def test_cost_from_parent(self, parent: "SquareObstacleAvgPlusWeightedGoal"):
        store_avg = self.avg_obstacle_min
        self.avg_obstacle_min = self._calculate_avg(parent.avg_obstacle_min, parent.generation, self.this_obstacle_min)
        cost = self._calculate_cost()
        self.avg_obstacle_min = store_avg
        return cost

    def _calculate_cost(self):
        return self.goal_dist * self.goal_weight + self.avg_obstacle_min

    @staticmethod
    def _calculate_avg(parent_avg, parent_gen, this_obstacle_min):
        """

        Parameters
        ----------
        parent_avg : float
            average obstacle minimum value of the parent heuristic
        parent_gen : int
            generation of the parent heuristic
        this_obstacle_min : float
            obstacle minimum value of this heuristic

        Returns
        -------
        float
            new average value using parent average and this obstacle min
        """
        prior_sum = parent_gen * parent_avg
        return (prior_sum + this_obstacle_min) / (parent_gen + 1)

    def get_cost(self):
        return self.cost
