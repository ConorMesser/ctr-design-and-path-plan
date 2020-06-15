import pathlib
import numpy as np
from time import strftime

from ctrdapp.config.parse_config import parse_config
from ctrdapp.collision.collision_checker import CollisionChecker
from ctrdapp.heuristic.heuristic_factory import create_heuristic_factory
from ctrdapp.optimize.optimizer_factory import create_optimizer


def main():
    import sys
    sys.setrecursionlimit(10 ** 4)  # todo global

    path = pathlib.Path().absolute()
    file = path / "configuration" / "config_ftl.yaml"
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

    # initial guess  todo
    initial_guess = [0.01] * (configuration.get('tube_number') * configuration.get('q_dof'))

    optimizer = create_optimizer(heuristic_factory, collision_detector,
                                 np.asarray(initial_guess), configuration)

    optimize_result = optimizer.find_min()

    # possibly rerun solver with a higher number of iterations
    # (especially for rrt* or other optimizing ones) todo

    optimize_result.save_result(output_path)

    # interactive?? todo
    optimize_result.best_solver.visualize_best_solution(objects_file, output_path)
    optimize_result.best_solver.visualize_best_solution_path(objects_file, output_path)


if __name__ == "__main__":
    main()
