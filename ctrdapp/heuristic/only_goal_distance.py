from .heuristic import Heuristic


class OnlyGoalDistance(Heuristic):

    def __init__(self, goal_dist):
        self.cost = goal_dist

    def get_cost(self):
        return self.cost

    def calculate_cost_from_parent(self, parent: "OnlyGoalDistance", reset=False):
        pass

    def test_cost_from_parent(self, parent: "OnlyGoalDistance"):
        print("Parent doesn't effect cost of Only Goal Distance heuristic.")
        return self.cost
