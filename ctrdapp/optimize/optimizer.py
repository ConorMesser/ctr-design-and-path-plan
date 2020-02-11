from abc import ABC, abstractmethod


class Optimizer(ABC):

    def __init__(self, heuristic_factory, collision_checker, initial_guess, configuration):
        self.tube_num = configuration.get("tube_number")
        self.precision = configuration.get("optimizer_precision")
        self.configuration = configuration
        self.heuristic_factory = heuristic_factory
        self.collision_checker = collision_checker
        self.initial_guess = initial_guess

    @abstractmethod
    def find_min(self):
        # should return q* for min cost and path (or maybe the best solver object)
        pass


