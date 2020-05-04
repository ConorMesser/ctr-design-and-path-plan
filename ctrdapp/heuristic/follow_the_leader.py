from .heuristic import Heuristic
import numpy as np


class FollowTheLeader(Heuristic):

    def __init__(self, only_tip, follow_the_leader):
        if only_tip:
            ftl_tip_array = follow_the_leader[-1][-1]
            self.ftl_magnitude = calculate_magnitude(ftl_tip_array)
        else:
            magnitudes = [calculate_magnitude(array) for tube in follow_the_leader for array in tube]
            self.ftl_magnitude = np.mean(magnitudes)

        self.avg_ftl = self.ftl_magnitude
        self.generation = 0
        self.cost = self._calculate_cost()

    def calculate_cost_from_parent(self, parent: 'FollowTheLeader', reset=False):
        if self.generation != 0 and not reset:
            print(f"Cost already calculated. Do not run method twice.")
            return
        self.avg_ftl = self._calculate_avg(parent.avg_ftl, parent.generation, self.ftl_magnitude)
        self.generation = parent.generation + 1
        self.cost = self._calculate_cost()

    def test_cost_from_parent(self, parent: "FollowTheLeader"):
        return self._calculate_avg(parent.avg_ftl, parent.generation, self.ftl_magnitude)

    def _calculate_cost(self):
        return self.avg_ftl

    @staticmethod
    def _calculate_avg(parent_avg, parent_gen, this_ftl):
        prior_sum = parent_gen * parent_avg
        return (prior_sum + this_ftl) / (parent_gen + 1)

    def get_cost(self):
        return self.cost


def calculate_magnitude(ftl_array):
    """ Calculate the magnitude of the array given (of omega and velocity)

    Calculation is based on the magnitude of a screw, where the magnitude
    equals the Euclidean norm of the omega unless ||omega|| = 0 in which case
    the magnitude equals the Euclidean norm of the velocity.

    Parameters
    ----------
    ftl_array : list (6x1 array) of floats
        omega and velocity values making up the follow-the-leader array

    Returns
    -------
    float : magnitude of input array
    """
    omega_norm = np.linalg.norm(ftl_array[0:3])
    vel = ftl_array[3:6]

    # Magnitude of a screw equals the Euclidean norm of the omega
    #  or the norm of the velocity (if omega_norm = 0)
    if omega_norm == 0:
        return np.linalg.norm(vel)
    else:
        return omega_norm


class FollowTheLeaderWInsertion(FollowTheLeader):

    def __init__(self, only_tip, insertion_weight, follow_the_leader, insertion_fraction):
        self.insertion_weight = insertion_weight
        self.insertion_fraction = insertion_fraction
        super().__init__(only_tip, follow_the_leader)

    def test_cost_from_parent(self, parent: "FollowTheLeaderWInsertion"):
        store_avg = self.avg_ftl
        self.avg_ftl = self._calculate_avg(parent.avg_ftl, parent.generation, self.ftl_magnitude)
        cost = self._calculate_cost()
        self.avg_ftl = store_avg
        return cost

    def _calculate_cost(self):
        return self.avg_ftl + self.insertion_weight * self.insertion_fraction
