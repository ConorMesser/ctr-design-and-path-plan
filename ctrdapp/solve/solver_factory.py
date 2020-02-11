from .solver import Solver
from .rrt import RRT
from .rrt_star import RRTStar


def create_solver(model_factory, heuristic_factory, collision_detector, configuration) -> Solver:
    name = configuration.get("solver_type")

    if name == "rrt":
        return RRT(model_factory, heuristic_factory, collision_detector, configuration)
    elif name == "rrt_star":
        return RRTStar(model_factory, heuristic_factory, collision_detector, configuration)
    else:
        raise UserWarning(f"{name} is not a defined solver. "
                          f"Change config file.")
