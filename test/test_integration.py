import pathlib
import unittest
import pyvista as pv
from math import pi

from ctrdapp.config.parse_config import parse_config
from ctrdapp.model.model import create_model
from ctrdapp.solve.visualize_utils import visualize_curve_single, add_single_curve, add_objects
from ctrdapp.heuristic.heuristic_factory import create_heuristic_factory
from ctrdapp.collision.collision_checker import CollisionChecker
from ctrdapp.solve.solver_factory import create_solver
from ctrdapp.heuristic.follow_the_leader import FollowTheLeader


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
        file = path / "configuration" / "config_ftl.yaml"
        configuration, dictionaries = parse_config(file)
        objects_file = path / "configuration" / configuration.get("collision_objects_filename")
        this_model = create_model(config=configuration, q=[[0.04]])

        # heuristic factory
        heuristic_factory = create_heuristic_factory(configuration,
                                                     dictionaries.get("heuristic"))
        # collision detector
        collision_detector = CollisionChecker(objects_file)

        # rrt
        this_solver = create_solver(this_model, heuristic_factory, collision_detector, configuration)

        # call get_best_cost
        cost, best_ind = this_solver.get_best_cost()
        print(cost)

        # this_solver.visualize_best_solution(objects_file)
        #
        # this_solver.visualize_best_solution_path(objects_file)
        this_solver.visualize_full_search()

        x = 5


    def test_visualize_solve_once(self):
        # create model
        path = pathlib.Path().absolute()
        file = path / "configuration" / "config_integration.yaml"
        configuration, dictionaries = parse_config(file)
        objects_file = path / "configuration" / configuration.get("collision_objects_filename")
        configuration["strain_bases"] = "linear, linear, quadratic"
        this_model = create_model(config=configuration, q=[[-0.02, 0.001], [0.03, 0.002]])

        # get g previous (using solve_g)
        insert_indices = [100, 100]
        prev_g = this_model.solve_g(indices=insert_indices)

        # try small step size + visualize
        delta_theta_s = [0.2, 0.5]
        delta_ins_s = [6, 5]
        this_ins = [4, 5]
        this_theta = [1, 1.2]
        g_out, eta_out, indices, true_insertion, ftl_heuristic = this_model.solve_integrate(delta_theta_s, delta_ins_s,
                                                                                            this_theta, this_ins, prev_g,
                                                                                            invert_insert=False)

        plotter = pv.Plotter()
        add_single_curve(plotter, g_out, 2, configuration.get("tube_radius"), None)
        add_objects(plotter, objects_file)
        plotter.show()
        # try large step size + visualize


# ----------- KEEP ------------
class TestModelAndHeuristics(unittest.TestCase):

    def setUp(self) -> None:
        path = pathlib.Path().absolute()
        file = path / "configuration" / "config_model_heuristic.yaml"
        self.configuration, dictionaries = parse_config(file)
        self.objects_file = path / "configuration" / "init_objects_blank.json"

        # single constant-curvature tube
        self.model_constant = create_model(config=self.configuration, q=[0.05])
        self.configuration['tube_number'] = 3
        self.configuration['tube_radius'] = [1.5, 1.2, 0.9]
        self.configuration['strain_bases'] = "constant, constant, constant"
        self.model_constant2 = create_model(config=self.configuration, q=[[pi/60], [0], [pi/60]])

    def testConstantModelFTL(self):
        for i in range(-5, 11, 4):  # regardless of the size of insertion
            for j in range(-2, 2):  # regardless of the theta value
                prev_g_out = self.model_constant.solve_g(indices=[60 + i * 2], thetas=[j], full=False)
                _, _, _, _, ftl_out = self.model_constant.solve_integrate([0], [i], [j], [30], prev_g_out)
                ftl_heuristic = FollowTheLeader(False, ftl_out)
                self.assertAlmostEqual(ftl_heuristic.get_cost(), 0, 14)

        prev_g_out = self.model_constant.solve_g(indices=[64], thetas=[-85], full=False)
        _, _, _, _, ftl_out = self.model_constant.solve_integrate([0.85], [2], [0], [30], prev_g_out)
        ftl_heuristic = FollowTheLeader(False, ftl_out)
        self.assertEqual(ftl_heuristic.get_cost(), 0.85)  # equal to the value of omega

    def testConstantModel2TubesFTL(self):
        # First and Last tubes have some constant curvature. Second is straight.
        # Initial configuration has last tube with 90 degrees of curvature.
        # First (largest) tube moves from no insertion to 2, but due to equal curvature with
        #  the last tube, the follow the leader heuristic for the last tube should be 0
        #  other than in the z-velocity (= -pi/3)
        prev_g_out = self.model_constant2.solve_g(indices=[120, 100, 60], thetas=[0, 0, 0], full=False)
        g_out, _, _, _, ftl_out = self.model_constant2.solve_integrate([0, 0, 0], [2, 0, 0], [0, 0, 0], [2, 10, 30], prev_g_out)

        # visualize_curve_single(g_out, self.objects_file,
        #                        self.configuration.get("tube_number"), self.configuration.get("tube_radius"))

        ftl_heuristic = FollowTheLeader(True, ftl_out)
        for i in range(0, 5):
            self.assertAlmostEqual(ftl_out[-1][-1][i], 0)
        self.assertAlmostEqual(ftl_out[-1][-1][-1], -pi/3)
        self.assertAlmostEqual(ftl_heuristic.get_cost(), pi/3)


if __name__ == '__main__':
    unittest.main()
