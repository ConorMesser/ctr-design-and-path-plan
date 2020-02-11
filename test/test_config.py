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
                            "tube_radius": 3,
                            "collision_objects_filename": "init_objects.json",
                            "optimizer_precision": 0.1,
                            "step_bound": 3,
                            "insertion_max": 120,
                            "nearest_neighbor_function": 2,
                            "iteration_number": 2000,
                            "goal_weight": 3,
                            "q_dof": 3,
                            "num_discrete_points": 101
                            }

    def test_parse_config(self):
        path = pathlib.Path().absolute()
        file = path / "configuration" / "config_specific_test.yaml"

        full_config, _ = parse_config(file)
        self.assertDictEqual(full_config, self.test_config)

        path = pathlib.Path().absolute()
        blank_file = path / "configuration" / "blank.yaml"

        blank_config, _ = parse_config(blank_file)
        default_keys = self.test_config.keys()
        blank_keys = blank_config.keys()
        # will fail if add'nl attributes are added or defaults changed****
        self.assertEqual(blank_keys, default_keys)


if __name__ == '__main__':
    unittest.main()
