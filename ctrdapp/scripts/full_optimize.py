import pathlib
import numpy as np
from time import strftime

from ctrdapp.config.parse_config import parse_config
from ctrdapp.collision.collision_checker import CollisionChecker
from ctrdapp.heuristic.heuristic_factory import create_heuristic_factory
from ctrdapp.optimize.optimizer_factory import create_optimizer


def full_optimize():
    import sys
    sys.setrecursionlimit(10 ** 4)

    path = pathlib.Path().absolute()
    config_filename = input("Enter name of configuration file (with .yaml extension): ")
    while True:
        file = path / "configuration" / config_filename
        if file.exists():
            break
        else:
            config_filename = input(f"{config_filename} doesn't exist. Please input configuration filename again: ")
    configuration, dictionaries = parse_config(file)

    run_name = configuration.get("run_identifier")
    output_path = path / "output" / run_name
    try:
        output_path.mkdir(parents=True)
    except FileExistsError:
        current_time = strftime("%H%M")
        current_date = strftime("%m%d%y")
        new_name = f"{run_name}_{current_date}_{current_time}"
        output_path = path / "output" / new_name
        output_path.mkdir(parents=True)

    objects_file = path / "configuration" / configuration.get("collision_objects_filename")
    collision_detector = CollisionChecker(objects_file)

    # create heuristic factory
    heuristic_factory = create_heuristic_factory(configuration,
                                                 dictionaries.get("heuristic"))

    # initial_guess = [0.01] * (configuration.get('tube_number') * configuration.get('q_dof'))
    # initial_guess = [[0.04, 0.02], [0.03, 0.001]]
    q_dof = configuration.get('q_dof')
    initial_guess_str = input(f"Enter initial guess for the q; should be {sum(q_dof)} "
                              f"numbers long, each separated by a comma. "
                              f"(Tubes have these degrees of freedom: {q_dof}).")

    while True:
        initial_guess = initial_guess_str.split(", ")
        initial_guess = [float(g) for g in initial_guess]
        if len(initial_guess) == sum(q_dof):
            break
        else:
            initial_guess_str = input(f"Incorrect format of initial guess. Input {sum(q_dof)} numbers "
                                      f"each separated by a comma: ")

    optimizer = create_optimizer(heuristic_factory, collision_detector,
                                 np.asarray(initial_guess), configuration)

    optimize_result = optimizer.find_min()

    # possibly rerun solver with a higher number of iterations
    # (especially for rrt* or other optimizing ones) todo

    optimize_result.save_result(output_path)

    # interactive?? todo
    optimize_result.best_solver.visualize_full_search(output_path, with_solution=True)
    optimize_result.best_solver.save_tree(output_path)
    optimize_result.best_solver.save_best_solution(output_path)
    optimize_result.best_solver.visualize_best_solution(objects_file, output_path)
    optimize_result.best_solver.visualize_best_solution_path(objects_file, output_path)


if __name__ == "__main__":
    full_optimize()
