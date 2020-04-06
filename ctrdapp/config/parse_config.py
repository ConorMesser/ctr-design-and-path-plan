import yaml


def parse_config(file):

    general = ["tube_number", "tube_radius", "collision_objects_filename"]
    optimizers = {"nelder_mead": ["optimizer_precision", "optimize_iterations"]}
    solvers = {"rrt": ["step_bound", "nearest_neighbor_function",
                       "iteration_number", "insertion_max",
                       "single_tube_control"],
               "rrt_star": ["step_bound", "nearest_neighbor_function",
                            "iteration_number", "rewire_probability",
                            "insertion_max", "single_tube_control"]}
    models = {"kinematic": ["q_dof", "num_discrete_points", "insertion_max"]}
    heuristics = {"square_obstacle_avg_plus_weighted_goal": ["goal_weight"],
                  "only_goal_distance": [],
                  "follow_the_leader": ["only_tip"]}

    dictionaries = {"optimizer": optimizers, "solver": solvers,
                    "model": models, "heuristic": heuristics}

    defaults = {"optimizer_type": "nelder_mead",
                "solver_type": "rrt",
                "model_type": "kinematic",
                "heuristic_type": "square_obstacle_avg_plus_weighted_goal",
                "tube_number": 2,
                "tube_radius": [3, 2],
                "collision_objects_filename": "init_objects.json",
                "optimizer_precision": 0.1,
                "step_bound": 3,
                "insertion_max": 100,
                "nearest_neighbor_function": 2,
                "iteration_number": 2000,
                "rewire_probability": 0.1,
                "goal_weight": 2,
                "q_dof": 3,
                "num_discrete_points": 101,
                "single_tube_control": True,
                "optimize_iterations": 50,
                "only_tip": True
                }

    with file.open(mode='r') as fid:
        config = yaml.full_load(fid)

    if config is None:  # for empty config file
        config = dict()

    for g in general:
        if g not in config:
            config[g] = defaults.get(g)
            print(f"{g} not specified in {file.name}. Using default value "
                  f"{defaults.get(g)} instead.")

    _config_helper("optimizer_type", optimizers, config, file.name, defaults)
    _config_helper("solver_type", solvers, config, file.name, defaults)
    _config_helper("model_type", models, config, file.name, defaults)
    _config_helper("heuristic_type", heuristics, config, file.name, defaults)

    return config, dictionaries


def _config_helper(config_key, required_dict, config, filename, defaults):
    this_type = config.get(config_key)
    if this_type not in required_dict:
        config[config_key] = defaults.get(config_key)
        print(f"{config_key} {this_type} does not exist. Using default of "
              f"{defaults.get(config_key)} instead.")

    new_type = config.get(config_key)
    for c in required_dict.get(new_type):
        if c not in config:
            config[c] = defaults.get(c)
            print(f"{config_key} is {new_type} however {c} not specified "
                  f"in {filename}. Default is {defaults.get(c)}.")
