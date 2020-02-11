from abc import ABC, abstractmethod
from .square_obstacle_avg_plus_weighted_goal import SquareObstacleAvgPlusWeightedGoal
from .only_goal_distance import OnlyGoalDistance
from .heuristic import Heuristic


class HeuristicFactory(ABC):

    @abstractmethod
    def factory_method(self, *args):
        pass

    def create(self, min_obstacle_distance, goal_distance) -> Heuristic:
        return self.factory_method(min_obstacle_distance, goal_distance)


class SOAPWGFactory(HeuristicFactory):

    def __init__(self, goal_weight, *args):
        self.goal_weight = goal_weight

    def factory_method(self, min_obstacle_distance, goal_distance) -> SquareObstacleAvgPlusWeightedGoal:
        return SquareObstacleAvgPlusWeightedGoal(self.goal_weight, min_obstacle_distance, goal_distance)


class OnlyGoalDistanceFactory(HeuristicFactory):

    def __init__(self, *args):
        pass

    def factory_method(self, min_obstacle_distance, goal_distance) -> OnlyGoalDistance:
        return OnlyGoalDistance(goal_distance)


def create_heuristic_factory(configuration: dict, heuristic_dict: dict) -> HeuristicFactory:
    name = configuration.get("heuristic_type")
    param_names = heuristic_dict.get(name)
    params = [configuration.get(n) for n in param_names]

    if name == "square_obstacle_avg_plus_weighted_goal":
        return SOAPWGFactory(*params)
    elif name == "only_goal_distance":
        return OnlyGoalDistanceFactory(*params)
    else:
        raise UserWarning(f"{name} is not a defined heuristic. "
                          f"Change config file.")
