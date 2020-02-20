import pathlib
import unittest
import math
import pyvista as pv

from ctrdapp.config.parse_config import parse_config
from ctrdapp.model.model import create_model, truncate_g
from ctrdapp.solve.visualize_utils import visualize_curve_single, add_single_curve, add_objects
from ctrdapp.heuristic.heuristic_factory import create_heuristic_factory
from ctrdapp.collision.collision_checker import CollisionChecker
from ctrdapp.solve.solver_factory import create_solver


class VisualizeUtilsTest(unittest.TestCase):

    def test_visualize(self):
        # need config
        path = pathlib.Path().absolute()
        file = path / "configuration" / "config_integration.yaml"
        configuration, dictionaries = parse_config(file)
        objects_file = path / "configuration" / configuration.get("collision_objects_filename")
        # need model
        this_model = create_model(config=configuration, q=[[0.01, 0.0005], [0.02, 0.0007]])

        # need to visualize
        g_out = this_model.solve_g(indices=[0, 0])

        visualize_curve_single(g_out, objects_file, configuration.get("tube_number"), configuration.get("tube_radius"))

    def test_RRT(self):
        path = pathlib.Path().absolute()
        file = path / "configuration" / "config_integration.yaml"
        configuration, dictionaries = parse_config(file)
        objects_file = path / "configuration" / configuration.get("collision_objects_filename")
        this_model = create_model(config=configuration, q=[[-0.01391], [0.02875]])

        # heuristic factory
        heuristic_factory = create_heuristic_factory(configuration,
                                                     dictionaries.get("heuristic"))
        # collision detector
        collision_detector = CollisionChecker(objects_file)

        # rrt
        this_solver = create_solver(this_model, heuristic_factory, collision_detector, configuration)

        # call get_best_cost
        cost, best_ind = this_solver.get_best_cost()

        this_solver.visualize_best_solution(objects_file)

        this_solver.visualize_best_solution_path(objects_file)


    def test_visualize_solve_once(self):
        # create model
        path = pathlib.Path().absolute()
        file = path / "configuration" / "config_integration.yaml"
        configuration, dictionaries = parse_config(file)
        objects_file = path / "configuration" / configuration.get("collision_objects_filename")
        configuration["strain_bases"] = "linear, linear, quadratic"
        this_model = create_model(config=configuration, q=[[-0.02, 0.001], [0.03, 0.002], [0.01, 0.0001]])

        # get g previous (using solve_g)
        insert_indices = [100, 100, 100]
        prev_g = this_model.solve_g(indices=insert_indices)

        # try small step size + visualize
        delta_theta_s = [0.2, 0.5, 0.4]
        delta_ins_s = [6, 5, 30]
        prev_ins = [10, 10, 10]
        g_out, eta_out, indices, true_insertion = this_model.solve_iterate(delta_theta_s, delta_ins_s, prev_ins, prev_g, invert_insert=False)

        plotter = pv.Plotter()
        g_trunc = truncate_g(g_out, indices)
        add_single_curve(plotter, g_trunc, 3, configuration.get("tube_radius"), None)
        add_objects(plotter, objects_file)
        plotter.show()
        # try large step size + visualize
