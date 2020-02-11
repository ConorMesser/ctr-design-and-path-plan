from abc import ABC, abstractmethod


class Heuristic(ABC):

    @abstractmethod
    def calculate_cost_from_parent(self, parent: "Heuristic"):
        pass

    @abstractmethod
    def get_cost(self):
        pass
