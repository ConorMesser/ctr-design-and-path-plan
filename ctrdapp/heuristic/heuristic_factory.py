from abc import ABC, abstractmethod
from .square_obstacle_avg_plus_weighted_goal import SquareObstacleAvgPlusWeightedGoal
from .only_goal_distance import OnlyGoalDistance
from .follow_the_leader import FollowTheLeader, FollowTheLeaderWInsertion
from .heuristic import Heuristic


class HeuristicFactory(ABC):

    @abstractmethod
    def factory_method(self, **kwargs):
        pass

    def create(self, **kwargs) -> Heuristic:
        return self.factory_method(**kwargs)


class SOAPWGFactory(HeuristicFactory):

    def __init__(self, goal_weight, *args):
        self.goal_weight = goal_weight

    def factory_method(self, **kwargs) -> SquareObstacleAvgPlusWeightedGoal:
        # uses inputs of min_obstacle_distance and goal_distance
        return SquareObstacleAvgPlusWeightedGoal(self.goal_weight, kwargs.get('min_obstacle_distance'), kwargs.get('goal_distance'))


class OnlyGoalDistanceFactory(HeuristicFactory):

    def __init__(self, *args):
        pass

    def factory_method(self, **kwargs) -> OnlyGoalDistance:
        # only uses goal_distance input
        return OnlyGoalDistance(kwargs.get('goal_distance'))


class FTLFactory(HeuristicFactory):

    def __init__(self, only_tip, *args):
        self.only_tip = only_tip

    def factory_method(self, **kwargs) -> FollowTheLeader:
        # only uses follow_the_leader input
        return FollowTheLeader(self.only_tip, kwargs.get('follow_the_leader'))


class FTLWInsertionFactory(FTLFactory):

    def __init__(self, only_tip, insertion_weight, *args):
        super().__init__(only_tip, *args)
        self.insertion_weight = insertion_weight

    def factory_method(self, **kwargs) -> FollowTheLeaderWInsertion:
        # only uses follow_the_leader input
        return FollowTheLeaderWInsertion(self.only_tip, self.insertion_weight,
                                         kwargs.get('follow_the_leader'),
                                         kwargs.get('insertion_fraction'))


def create_heuristic_factory(configuration: dict, heuristic_dict: dict) -> HeuristicFactory:
    name = configuration.get("heuristic_type")
    param_names = heuristic_dict.get(name)
    params = [configuration.get(n) for n in param_names]

    if name == "square_obstacle_avg_plus_weighted_goal":
        return SOAPWGFactory(*params)
    elif name == "only_goal_distance":
        return OnlyGoalDistanceFactory(*params)
    elif name == "follow_the_leader":
        return FTLFactory(*params)
    elif name == "follow_the_leader_w_insertion":
        return FTLWInsertionFactory(*params)
    else:
        raise UserWarning(f"{name} is not a defined heuristic. "
                          f"Change config file.")
