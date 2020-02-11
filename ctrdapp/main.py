import pathlib
import numpy as np

from .config.parse_config import parse_config
from .collision.collision_checker import CollisionChecker
from .heuristic.heuristic_factory import create_heuristic_factory
from .optimize.optimizer_factory import create_optimizer


def main():

    path = pathlib.Path().absolute()
    file = path / "configuration" / "config.yaml"
    configuration, dictionaries = parse_config(file)
    objects_file = path / "configuration" / configuration.get("collision_objects_filename")
    collision_detector = CollisionChecker(objects_file)

    # create model factory  todo is this necessary?

    # create heuristic factory
    heuristic_factory = create_heuristic_factory(configuration,
                                                 dictionaries.get("heuristic"))

    # initial guess  todo
    initial_guess = [1] * (configuration.get('tube_number') * configuration.get('q_dof'))

    optimizer = create_optimizer(heuristic_factory, collision_detector,
                                 np.asarray(initial_guess), configuration)

    optimizer.find_min()  # outputs what? todo
