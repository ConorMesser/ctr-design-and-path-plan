import unittest
import pathlib

from ctrdapp.config.parse_config import parse_config


class TestParseConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.test_config = {"optimizer_type": "nelder_mead",
                            "solver_type": "rrt",
                            "model_type": "kinematic",
                            "heuristic_type": "square_obstacle_avg_plus_weighted_goal",
                            "tube_number": 2,
                            "tube_lengths": [60, 50],
                            "tube_radius": {'innner': [2.9, 1.9], 'outer': [3, 2]},
                            "collision_objects_filename": "init_objects.json",
                            "optimizer_precision": 0.1,
                            "optimize_iterations": 50,
                            "step_bound": 3,
                            "delta_x": 1,
                            "iteration_number": 2000,
                            "goal_weight": 2,
                            "q_dof": [1, 1],
                            "single_tube_control": True,
                            "strain_bases": ["constant", "constant"]
                            }

    def test_parse_config(self):  # todo update to test config validation
        path = pathlib.Path().absolute()
        file = path / "configuration" / "config_specific_test.yaml"

        full_config, _ = parse_config(file)
        self.assertDictEqual(full_config, self.test_config)

        path = pathlib.Path().absolute()
        blank_file = path / "configuration" / "blank.yaml"

        blank_config, _ = parse_config(blank_file)
        default_keys = self.test_config.keys()  # default for specified RRT, kinematic, SOAPWG, Nelder-Mead
        blank_keys = blank_config.keys()
        # will fail if add'nl attributes are added or defaults changed****
        self.assertCountEqual(blank_keys, default_keys)


if __name__ == '__main__':
    unittest.main()
