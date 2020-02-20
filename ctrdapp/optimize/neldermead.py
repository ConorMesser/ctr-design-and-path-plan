from scipy import optimize
import numpy as np

from .optimizer import Optimizer
from ..model.model import create_model
from ..solve.solver_factory import create_solver


class NelderMead(Optimizer):

    def __init__(self, heuristic_factory, collision_checker, initial_guess, configuration):
        super().__init__(heuristic_factory, collision_checker, initial_guess, configuration)

        self.solver_store = []
        self.costs = []

    def find_min(self):
        func = self.solver_heuristic
        init_simplex = np.array([[0.02, 0.0002, 0.02, 0.0002],
                                 [0.01, 0.0002, 0.02, 0.0006],
                                 [0.02, -0.0005, 0.04, 0.0002],
                                 [0.015, 0.0002, 0.01, 0.0002],
                                 [0.02, 0.0004, 0.02, -0.0004]])  # todo
        # init_simplex = np.array([[0.02, 0.02],
        #                          [0.005, 0.03],
        #                          [0.01, 0.015]])

        # Q should be bounded (in both + and -) by (2*Emax)/Tube Diameter
        # This is due to physical constraints on the bending curvature (nitinol
        # must remain in the elastic/plastic region given by the Emax)
        # The constant and changing q parameters must be considered to limit the
        # max q possible

        optimize_result = optimize.minimize(func, self.initial_guess,
                                            method='Nelder-Mead',
                                            options={'xatol': self.precision,
                                                     'fatol': self.precision,
                                                     'initial_simplex': init_simplex,
                                                     'maxfev': self.configuration.get('optimize_iterations')})

        best_array = optimize_result.x
        # best_score = optimize_result.fun
        # all_q = optimize_result.allvecs

        best_solver = self._find_solver(best_array)

        return best_solver, self.solver_store

    def solver_heuristic(self, x: [float]) -> float:

        # model from x
        this_model = create_model(self.configuration, x)

        # model, heuristic_factory, collision_detector, configuration
        this_solver = create_solver(this_model, self.heuristic_factory, self.collision_checker, self.configuration)

        # call get_best_cost
        cost, index = this_solver.get_best_cost()

        solver_tuple = {'cost': cost, 'solver': this_solver}

        self._insert_tuple(solver_tuple)

        # return cost
        return cost

    def _find_solver(self, array):
        for s in self.solver_store:
            this_q = s.get('solver').model.q
            if (this_q.flatten() == array.flatten()).all():
                return s
        print(f"Could not find desired solver for q = {array}. "
              f"Giving last solver instead.")
        return self.solver_store[0]

    def _insert_tuple(self, solver_tuple):
        list_length = len(self.solver_store)
        for ind, s in enumerate(self.solver_store):
            if solver_tuple.get('cost') < s.get('cost'):
                self.solver_store.insert(ind, solver_tuple)
                break
        if len(self.solver_store) == list_length:
            self.solver_store.append(solver_tuple)
        if len(self.solver_store) > 7:
            self.solver_store.pop()
