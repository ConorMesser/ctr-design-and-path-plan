"""Parses a given configuration file."""

import yaml


def parse_config(file):
    """Parses given yaml file using requirements and defaults.

    Required parameters are defined in this function along with
    parameter defaults. Parameters are grouped by package name.

    Parameters
    ----------
    file : pathlib.PosixPath
        configuration file

    Returns
    -------
    (dict, list[dict])
        dictionary of configuration parameters;
        list of dictionaries of key-names

    Notes
    -----
    This file must be kept up to date as new modules are added to
    different packages as well as new parameters being specified. To add
    a new parameter, add its name to the appropriate dictionary and to
    the defaults dictionary (with a suitable default. To add a
    new module (new type of solver, optimizer, model, or heuristic),
    add a new entry to the dictionaries below with a list of all
    the required parameters.
    """

    # dictionaries specifying required parameters for various modules
    general = ["tube_number", "tube_radius", "collision_objects_filename"]
    optimizers = {"nelder_mead": ["optimizer_precision", "optimize_iterations"]}
    solvers = {"rrt": ["step_bound", "iteration_number",
                       "insertion_max", "single_tube_control"],
               "rrt_star": ["step_bound", "iteration_number", "insertion_max"]}
    models = {"kinematic": ["q_dof", "num_discrete_points", "insertion_max"]}
    heuristics = {"square_obstacle_avg_plus_weighted_goal": ["goal_weight"],
                  "only_goal_distance": [],
                  "follow_the_leader": ["only_tip"],
                  "follow_the_leader_w_insertion": ["only_tip", "insertion_weight"]}

    # groups together the required parameter dictionaries
    dictionaries = {"optimizer": optimizers, "solver": solvers,
                    "model": models, "heuristic": heuristics}

    # dictionary detailing all of the default parameter values
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
                "iteration_number": 2000,
                "rewire_probability": 0.1,
                "goal_weight": 2,
                "q_dof": 3,
                "num_discrete_points": 101,
                "single_tube_control": True,
                "optimize_iterations": 50,
                "only_tip": True,
                "insertion_weight": 10
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
    """Helper function that sets parameters for given package type.

    Gets the list of required parameters for the module specified by the
    config_key entry in the config. Adds these parameters to the config,
    if they exist; if not, sets to the default values with a printed
    message detailing the omission.

    Parameters
    ----------
    config_key : str
        describes a package as one of four options: "optimizer_type",
        "solver_type", "model_type", "heuristic_type"
    required_dict : dict
        dictionary of required parameters for package described in config_key
    config : dict
        current dictionary of configuration parameters
    filename : str
        name of configuration file for error-message purposes
    defaults : dict
        dictionary with all the default parameter values

    Returns
    -------
    None
    """
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
