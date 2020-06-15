from scipy import optimize
import numpy as np
import time

from .optimizer import Optimizer
from ..model.kinematic import create_model
from ..solve.solver_factory import create_solver
from ctrdapp.model.strain_bases import max_from_base
from .optimize_result import OptimizeResult


class NelderMead(Optimizer):

    def __init__(self, heuristic_factory, collision_checker, initial_guess, configuration):
        super().__init__(heuristic_factory, collision_checker, initial_guess, configuration)

        self.solver_results = []
        self.best_solver = {'cost': 100000000, 'solver': None}
        self.costs = []

    def find_min(self):
        func = self.solver_heuristic

        init_simplex = calculate_simplex(self.configuration.get('strain_bases'),
                                         self.configuration.get('tube_radius'),
                                         self.configuration.get('insertion_max'),
                                         self.configuration.get('q_dof'))
        print(init_simplex)
        # alter_simplex = input("Would you like to alter the simplex? (yes/no): ")
        # if alter_simplex == 'yes':
        #     new_simplex = input("Write updated simplex: ")
        #     same_size = True
        #     for i, arr in enumerate(init_simplex):
        #         same_size = same_size and len(arr) == len(new_simplex[i])
        #     if same_size:
        #         init_simplex = alter_simplex
        #     else:
        #         print('Input was not the correct size. Using original simplex.')

        start_time = time.time()

        optimize_result_nm = optimize.minimize(func, self.initial_guess,
                                               method='Nelder-Mead',
                                               options={'xatol': self.precision,
                                                        'fatol': self.precision,
                                                        'initial_simplex': init_simplex,
                                                        'maxfev': self.configuration.get('optimize_iterations')})
        end_time = time.time()

        optimize_result = OptimizeResult(optimize_result_nm.x,
                                         self.best_solver.get('solver'),
                                         optimize_result_nm.success,
                                         optimize_result_nm.nfev,
                                         optimize_result_nm.nit,
                                         end_time - start_time,
                                         self.solver_results)

        return optimize_result

    def solver_heuristic(self, x: [float]) -> float:

        # model from x
        this_model = create_model(self.configuration, x)

        this_model.solve_g()
        # model, heuristic_factory, collision_detector, configuration
        this_solver = create_solver(this_model, self.heuristic_factory, self.collision_checker, self.configuration)

        # call get_best_cost
        cost, index = this_solver.get_best_cost()

        solver_dict = {'q': this_solver.model.q, 'cost': cost}
        self.solver_results.append(solver_dict)

        if cost < self.best_solver.get('cost'):
            self.best_solver = {'cost': cost, 'solver': this_solver}

        return cost

    def _find_solver(self, array):
        for s in self.solver_results:
            this_q = s.get('q')
            if (this_q.flatten() == array.flatten()).all():
                return s
        print(f"Could not find desired solver for q = {array}. "
              f"Giving last solver instead.")
        return self.solver_results[-1]


# Q should be bounded (in both + and -) by (2*Emax)/Tube Diameter
# This is due to physical constraints on the bending curvature (nitinol
# must remain in the elastic/plastic region given by the Emax)
# The constant and changing q parameters must be considered to limit the
# max q possible
def calculate_simplex(base_names, tube_radii, length, q_dof, e_max=0.05):

    max_init_array = []
    for base, tube_rad in zip(base_names, tube_radii):
        tube_arr = max_from_base(base, e_max / tube_rad, length, q_dof)
        max_init_array.extend(tube_arr)
    dim = len(max_init_array)
    init_simplex = [[q / 2 for q in max_init_array]]
    for i in range(dim):
        new_array = max_init_array.copy()
        new_array[i] = new_array[i] / 4
        init_simplex.append(new_array)

    return init_simplex
