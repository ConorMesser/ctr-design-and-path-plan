from .optimizer import Optimizer
from .neldermead import NelderMead


def create_optimizer(heuristic_factory, collision_checker, initial_guess, configuration) -> Optimizer:
    name = configuration.get("optimizer_type")

    if name == "nelder_mead":
        return NelderMead(heuristic_factory, collision_checker, initial_guess, configuration)
    else:
        raise UserWarning(f"{name} is not a defined optimizer. "
                          f"Change config file.")
