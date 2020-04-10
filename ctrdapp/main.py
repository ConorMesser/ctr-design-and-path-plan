import pathlib
import numpy as np

from ctrdapp.config.parse_config import parse_config
from ctrdapp.collision.collision_checker import CollisionChecker
from ctrdapp.heuristic.heuristic_factory import create_heuristic_factory
from ctrdapp.optimize.optimizer_factory import create_optimizer


def main():

    path = pathlib.Path().absolute()
    file = path / "configuration" / "config_trial.yaml"
    configuration, dictionaries = parse_config(file)
    objects_file = path / "configuration" / configuration.get("collision_objects_filename")
    collision_detector = CollisionChecker(objects_file)

    # create heuristic factory
    heuristic_factory = create_heuristic_factory(configuration,
                                                 dictionaries.get("heuristic"))

    # initial guess  todo
    initial_guess = [0.01] * (configuration.get('tube_number') * configuration.get('q_dof'))

    optimizer = create_optimizer(heuristic_factory, collision_detector,
                                 np.asarray(initial_guess), configuration)

    best_solver, solvers = optimizer.find_min()

    # possibly rerun solver with a higher number of iterations
    # (especially for rrt* or other optimizing ones) todo

    solution_cost, solution_index = best_solver.get('solver').get_best_cost()
    solution_path = best_solver.get('solver').get_path(solution_index)
    # save path, best q, and cost todo

    # interactive?? todo
    best_solver.get('solver').visualize_best_solution(objects_file)  # save picture todo
    best_solver.get('solver').visualize_best_solution_path(objects_file)  # save movie todo


if __name__ == "__main__":
    main()
