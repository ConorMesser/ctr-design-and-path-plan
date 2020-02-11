from scipy import optimize

from .optimizer import Optimizer
from ..model.model import create_model
from ..solve.solver_factory import create_solver


class NelderMead(Optimizer):

    def __init__(self, heuristic_factory, collision_checker, initial_guess, configuration):
        super().__init__(heuristic_factory, collision_checker, initial_guess, configuration)

        self.solver_store = []

    def find_min(self):
        func = self.solver_heuristic
        best_array, best_score = optimize.minimize(func, self.initial_guess,
                                                   method='Nelder-Mead',
                                                   options={'xatol': self.precision, 'fatol': self.precision})

        return best_array, best_score

    def solver_heuristic(self, x: [float]) -> float:

        # model from x
        this_model = create_model(self.configuration, x)

        # model, heuristic_factory, collision_detector, configuration
        this_solver = create_solver(this_model, self.heuristic_factory, self.collision_checker, self.configuration)

        # call get_best_cost
        cost = this_solver.get_best_cost()

        # store this solver/model??? todo
        self.solver_store.append(this_solver)

        # return cost
        return cost
