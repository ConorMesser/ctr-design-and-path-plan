import unittest
import numpy as np
import math
import timeit

from ctrdapp.model import matrix_utils
from ctrdapp.model.kinematic import Kinematic, calculate_indices
from ctrdapp.model.strain_bases import linear_strain, constant_strain


class TestMatrixUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.vec3 = np.array([1, 2, 3])
        self.vec6 = np.array([1, 2, 3, 4, 5, 6])
        self.tilde_out = np.array([[0, -3, 2],
                                   [3, 0, -1],
                                   [-2, 1, 0]])
        self.hat_out = np.array([[0, -3, 2, 4],
                                 [3, 0, -1, 5],
                                 [-2, 1, 0, 6],
                                 [0, 0, 0, 0]])
        self.adjoint = np.array([[0, -3, 2, 0, 0, 0],
                                 [3, 0, -1, 0, 0, 0],
                                 [-2, 1, 0, 0, 0, 0],
                                 [-28, 5, 6, 0, -3, 2],
                                 [8, -22, 12, 3, 0, -1],
                                 [12, 15, -14, -2, 1, 0]])
        self.little_adjoint = np.array([[0, -3, 2, 0, 0, 0],
                                        [3, 0, -1, 0, 0, 0],
                                        [-2, 1, 0, 0, 0, 0],
                                        [0, -6, 5, 0, -3, 2],
                                        [6, -0, -4, 3, 0, -1],
                                        [-5, 4, 0, -2, 1, 0]])
        self.exp_map_0 = np.array([[1, -3, 2, 4],
                                   [3, 1, -1, 5],
                                   [-2, 1, 1, 6],
                                   [0, 0, 0, 1]])
        self.exp_map_pi = np.array([[-1.6344, 1.6608, -0.2291, 0.9604],
                                   [-0.8502, -1.0264, 1.6344, 5.6079],
                                   [1.4449, 0.7974, -0.0132, 6.6079],
                                   [0, 0, 0, 1]])
        self.t_exp_0 = np.array([[1, -1.5, 1, 0, 0, 0],
                                [1.5, 1, -0.5, 0, 0, 0],
                                [-1, 0.5, 1, 0, 0, 0],
                                [0, -3, 2.5, 1, -1.5, 1],
                                [3, 0, -2, 1.5, 1, -0.5],
                                [-2.5, 2, 0, -1, 0.5, 1]])
        self.t_exp_pi = np.array([[-0.0416, -0.1933, 0.476, 0, 0, 0],
                                 [0.5138, 0.1988, 0.3629, 0, 0, 0],
                                 [0.0047, 0.5986, 0.5994, 0, 0, 0],
                                 [-0.2161, 3.6196, -1.5823, -0.0416, -0.1933, 0.476],
                                 [-2.8506, -0.2402, 1.0349, 0.5138, 0.1988, 0.3629],
                                 [2.4955, -0.6505, -0.6008, 0.0047, 0.5986, 0.5994]])

    def test_big_adjoint(self):
        np.testing.assert_equal(matrix_utils.big_adjoint(self.hat_out),
                                self.adjoint)

    def test_little_adjoint(self):
        np.testing.assert_equal(matrix_utils.little_adjoint(self.vec6),
                                self.little_adjoint)
        np.testing.assert_equal(matrix_utils.little_adjoint([1, 2, 3, 4, 5, 6]),
                                self.little_adjoint)

    def test_dynamic_tilde(self):
        np.testing.assert_equal(matrix_utils.tilde(self.vec3),
                                self.tilde_out)
        np.testing.assert_equal(matrix_utils.tilde([1, 2, 3]),
                                self.tilde_out)

    def test_dynamic_hat(self):
        np.testing.assert_equal(matrix_utils.hat(self.vec6),
                                self.hat_out)
        np.testing.assert_equal(matrix_utils.hat([1, 2, 3, 4, 5, 6]),
                                self.hat_out)

    def test_exponential_map(self):
        np.testing.assert_equal(matrix_utils.exponential_map(0, self.hat_out),
                                self.exp_map_0)
        np.testing.assert_almost_equal(matrix_utils.exponential_map(math.pi, self.hat_out),
                                       self.exp_map_pi, decimal=3)

    def test_t_exponential(self):
        np.testing.assert_equal(matrix_utils.t_exponential(1, 0, self.vec6),
                                self.t_exp_0)
        np.testing.assert_almost_equal(matrix_utils.t_exponential(1, math.pi, self.vec6),
                                       self.t_exp_pi, decimal=3)


class TestKinematic(unittest.TestCase):
    def setUp(self) -> None:
        strain_bias = np.array([0, 0, 0, 1, 0, 0])

        no_strain_q1 = [np.array([0, 0])]
        no_strain_q2 = [np.array([0, 0]), np.array([0, 0])]
        constant_q1 = [np.array([1, 0])]
        constant_q2 = [np.array([1, 0]), np.array([1, 0])]
        no_strain_1_constant_2 = [np.array([0, 0]), np.array([1, 0])]

        base1 = [linear_strain]
        base2 = [linear_strain, linear_strain]

        self.no_strain1 = Kinematic(1, no_strain_q1, [2], [1], 0.1, base1)
        self.no_strain1_rough = Kinematic(1, no_strain_q1, [2], [1], 0.5, base1)
        self.no_strain1_fine = Kinematic(1, no_strain_q1, [2], [1], 0.05, base1)
        self.no_strain2 = Kinematic(2, no_strain_q2, [2, 2], [1, 1], 0.1, base2)
        self.constant_strain1 = Kinematic(1, constant_q1, [2], [math.pi / 2], math.pi / 20, base1)
        self.constant_strain1_fine = Kinematic(1, constant_q1, [2], [math.pi / 2], math.pi / 40, base1)
        self.constant_strain2 = Kinematic(2, constant_q2, [2, 2], [math.pi / 2, math.pi / 2], math.pi / 20, base2)
        self.combo = Kinematic(2, no_strain_1_constant_2, [2, 2], [1, 1], 11, base2)

        self.constant_tip = np.array([[0, 0, 1, 1],
                                      [0, 1, 0, 0],
                                      [-1, 0, 0, -1],
                                      [0, 0, 0, 1]])

    def test_solve_g_one_tube_no_strain(self):

        # Initial insertion
        no_strain1_g = self.no_strain1.solve_g(full=True)[0]
        no_strain1_g_disc = self.no_strain1_fine.solve_g(full=True)[0]
        eye_end = np.eye(4)
        eye_end[0, 3] = -1

        np.testing.assert_equal(no_strain1_g[-1], np.eye(4))
        np.testing.assert_almost_equal(no_strain1_g[0], eye_end)
        for i in range(11):
            np.testing.assert_almost_equal(no_strain1_g[i], no_strain1_g_disc[i*2])

        # Half insertion
        no_strain1_g = self.no_strain1.solve_g(indices=[5], full=True)[0]
        no_strain1_g_disc = self.no_strain1_fine.solve_g(indices=[10], full=True)[0]
        eye_end[0, 3] = -0.5

        np.testing.assert_equal(no_strain1_g[5], np.eye(4))
        np.testing.assert_almost_equal(no_strain1_g[0], eye_end)
        for i in range(11):
            np.testing.assert_almost_equal(no_strain1_g[i], no_strain1_g_disc[i * 2])

        # Full insertion
        no_strain1_g = self.no_strain1.solve_g(indices=[0], full=True)[0]
        no_strain1_g_disc = self.no_strain1_fine.solve_g(indices=[0], full=True)[0]
        eye_end[0, 3] = 1

        np.testing.assert_equal(no_strain1_g[0], np.eye(4))
        np.testing.assert_almost_equal(no_strain1_g[-1], eye_end)
        for i in range(11):
            np.testing.assert_almost_equal(no_strain1_g[i], no_strain1_g_disc[i * 2])

        # Theta tests
        no_strain1_g_rot_2pi = self.no_strain1.solve_g(indices=[0], thetas=[2*math.pi], full=True)[0]
        no_strain1_g_rot_halfpi = self.no_strain1.solve_g(thetas=[math.pi/2], full=True)[0]
        eye_90 = np.eye(4)
        eye_90[1:3, 1:3] = [[0, -1], [1, 0]]

        np.testing.assert_equal(no_strain1_g_rot_2pi[0], np.eye(4))
        np.testing.assert_almost_equal(no_strain1_g_rot_halfpi[-1], eye_90)
        for i in range(11):
            np.testing.assert_almost_equal(no_strain1_g[i], no_strain1_g_rot_2pi[i])

    def test_solve_g_one_tube_constant_strain(self):
        constant_strain1_g = self.constant_strain1.solve_g(indices=[0], full=True)[0]
        constant_strain1_g_fine = self.constant_strain1_fine.solve_g(indices=[0], full=True)[0]

        np.testing.assert_equal(constant_strain1_g[0], np.eye(4))
        np.testing.assert_almost_equal(constant_strain1_g[-1], self.constant_tip)
        for i in range(11):
            np.testing.assert_almost_equal(constant_strain1_g[i], constant_strain1_g_fine[i * 2])

    def test_solve_g_two_tubes_nostrain(self):
        no_strain2_g = self.no_strain2.solve_g(full=True)
        g_first_extended = self.no_strain2.solve_g(indices=[0, 10], full=True)

        for i in range(11):
            np.testing.assert_equal(no_strain2_g[0][i], no_strain2_g[1][i])
            np.testing.assert_almost_equal(g_first_extended[0][i], g_first_extended[1][i])

        no_strain2_g = self.no_strain2.solve_g(indices=[0, 0], full=True)
        np.testing.assert_equal(no_strain2_g[0][0], np.eye(4))
        np.testing.assert_equal(no_strain2_g[0][-1], no_strain2_g[1][0])

        end_g = np.eye(4)
        end_g[0, 3] = 2
        np.testing.assert_almost_equal(no_strain2_g[1][-1], end_g)

    def test_solve_g_two_tubes_constant(self):
        this_g = self.constant_strain2.solve_g(indices=[0, 0], full=True)
        turn_g = self.constant_strain2.solve_g(indices=[0, 0], thetas=[0, math.pi/2], full=True)

        np.testing.assert_equal(this_g[0][0], np.eye(4))
        np.testing.assert_almost_equal(this_g[0][-1], self.constant_tip)
        np.testing.assert_almost_equal(this_g[0][-1], this_g[1][0])

        final_tip = np.array([[-1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, -1, -2],
                              [0, 0, 0, 1]])
        final_tip_turn = np.array([[0, 1, 0, 1],
                                  [1, 0, 0, 1],
                                  [0, 0, -1, -2],
                                  [0, 0, 0, 1]])

        np.testing.assert_almost_equal(this_g[1][-1], final_tip)
        np.testing.assert_almost_equal(turn_g[1][-1], final_tip_turn)

    # Inverted s and intuitive
    # Test insertion values (<, >= tube length, 0)
    # Test velocity (pos, insertion neg; neg, insertion pos)
    # Check size of eta_out (q_dot bool)
    # Check g and eta_out
    # - Should be very simple, giving simple g_previous
    # Testing for q_dot much harder
    def test_solve_once(self):  # todo
        a = np.eye(4)
        b = np.eye(4)
        b[0, 3] = 0.5
        c = np.eye(4)
        c[0, 3] = 1
        insert_none = [[a]]
        insert_half = [[a, b]]
        insert_full = [[a, b, c]]

        # only one eta value because no q_dot
        eta_none = np.array([[[0, 0, 0, 0, 0, 0]]])
        eta_forward_half = np.array([[[0, 0, 0, 0.5, 0, 0]]])  # "forward" is insertion; velocity = -delta
        eta_backwards_half = np.array([[[0, 0, 0, -0.5, 0, 0]]])  # "backwards" is retraction; velocity = -delta

        # zero insertion
        g_test, eta_test, insert_indices, true_insertions, ftl_heuristic = self.no_strain1_rough.solve_integrate([0],
                                                                                                                 [0],
                                                                                                                 [0],
                                                                                                                 [0],
                                                                                                                 insert_none,
                                                                                                                 invert_insert=True)
        np.testing.assert_equal(g_test, insert_none)
        np.testing.assert_equal(true_insertions, [0])
        np.testing.assert_equal(insert_indices, [2])
        np.testing.assert_equal(eta_test, eta_none)
        np.testing.assert_equal(ftl_heuristic, [[0]])

        # Round away from zero tests
        g_test_forward, eta_test, insert_indices, true_insertions, ftl_heuristic = \
            self.no_strain1_rough.solve_integrate([0], [0.499], [0], [0.499], insert_none, invert_insert=True)
        np.testing.assert_equal(g_test_forward, insert_half)
        np.testing.assert_equal(true_insertions, [0.5])
        np.testing.assert_equal(insert_indices, [1])
        np.testing.assert_almost_equal(eta_test, eta_forward_half, 3)
        np.testing.assert_equal(ftl_heuristic, [[0]])

        g_test_backward, eta_test, insert_indices, true_insertions, ftl_heuristic = \
            self.no_strain1_rough.solve_integrate([0], [-0.499], [0], [0.501], insert_none, invert_insert=False)
        np.testing.assert_equal(g_test_backward, insert_half)
        np.testing.assert_equal(true_insertions, [0.5])
        np.testing.assert_equal(insert_indices, [1])
        np.testing.assert_almost_equal(eta_test, eta_forward_half, 3)
        np.testing.assert_equal(ftl_heuristic, [[0]])

        g_test_forward, eta_test, insert_indices, true_insertions, ftl_heuristic = \
            self.no_strain1_rough.solve_integrate([0], [0.501], [0], [0.501], insert_none, invert_insert=True)
        np.testing.assert_equal(g_test_forward, insert_half)
        np.testing.assert_equal(true_insertions, [0.5])
        np.testing.assert_equal(insert_indices, [1])
        np.testing.assert_almost_equal(eta_test, eta_forward_half, 3)
        np.testing.assert_equal(ftl_heuristic, [[0]])

        g_test_backward, eta_test, insert_indices, true_insertions, ftl_heuristic = \
            self.no_strain1_rough.solve_integrate([0], [-0.501], [0], [0.499], insert_none, invert_insert=False)
        np.testing.assert_equal(g_test_backward, insert_half)
        np.testing.assert_equal(true_insertions, [0.5])
        np.testing.assert_equal(insert_indices, [1])
        np.testing.assert_almost_equal(eta_test, eta_forward_half, 3)
        np.testing.assert_equal(ftl_heuristic, [[0]])

        # Don't allow retraction past previous tube
        g_test, eta_test, insert_indices, true_insertions, ftl_heuristic = self.no_strain1_rough.solve_integrate([0],
                                                                                                                 [-0.1],
                                                                                                                 [0], [
                                                                                                                     -0.101],
                                                                                                                 insert_none,
                                                                                                                 invert_insert=True)
        np.testing.assert_equal(g_test, insert_none)
        np.testing.assert_equal(true_insertions, [0])
        np.testing.assert_equal(insert_indices, [2])
        np.testing.assert_equal(eta_test, np.array([[[0, 0, 0, -0.1, 0, 0]]]))  # velocity remains despite input error
        np.testing.assert_equal(ftl_heuristic, [[0]])

        # Don't allow extension past max tube length
        this_g = self.no_strain1_rough.solve_g([2], [0], full=True)
        g_test_forward, eta_test, insert_indices, true_insertions, ftl_heuristic = \
            self.no_strain1_rough.solve_integrate([0], [0.1], [0], [1.101], insert_full, invert_insert=True)
        np.testing.assert_equal(g_test_forward, insert_full)
        np.testing.assert_equal(true_insertions, [1])
        np.testing.assert_equal(insert_indices, [0])
        np.testing.assert_equal(eta_test, np.array([[[0, 0, 0, 0.1, 0, 0]]]))
        np.testing.assert_equal(ftl_heuristic, [[0, 0, 0]])

    def test_solve_once_speed(self):
        # zero insertion
        iter_num = 5000
        this_g = self.no_strain1_rough.solve_g([2], [0], full=True)
        time = timeit.Timer(lambda: self.no_strain1_rough.solve_integrate([0], [-0.501], [1], [0.499], this_g,
                                                                          invert_insert=False)).timeit(iter_num)
        self.assertLess(time/iter_num, 0.0004)

    def test_solve_g_speed(self):
        iter_num = 500
        time_full = timeit.Timer(lambda: self.constant_strain2.solve_g(
            indices=[0, 0], thetas=[0, math.pi/2], full=True)).timeit(iter_num)
        self.assertLess(time_full/iter_num, 0.005)

        time_part = timeit.Timer(lambda: self.constant_strain2.solve_g(
            indices=[0, 0], thetas=[0, math.pi / 2], full=False)).timeit(iter_num)
        self.assertAlmostEqual(time_part/iter_num, time_full/iter_num, 3)

        time_full_no_extension = timeit.Timer(lambda: self.constant_strain2.solve_g(
            thetas=[0, math.pi/2], full=True)).timeit(iter_num)
        time_part_no_extension = timeit.Timer(lambda: self.constant_strain2.solve_g(
            thetas=[0, math.pi / 2], full=False)).timeit(iter_num)
        self.assertLess(time_part_no_extension, time_full_no_extension)
        self.assertLess(time_full, time_full_no_extension)

    def test_FTL_constant_curvature(self):
        strain_bias = np.array([0, 0, 0, 1, 0, 0])
        strain_base = [constant_strain]

        model = Kinematic(1, [np.array([0.05])], [1], [50], 0.5, strain_base)
        prev_g_out = model.solve_g([38], [0], full=False)
        g_out = model.solve_g([36], [0], full=False)

        _, ftl_out = model.solve_eta([-2], [38], [0], prev_g_out, g_out)
        ftl_averages = [np.mean(array) for tube in ftl_out for array in tube]

        tube_length = len(ftl_out[-1])
        for i in [0, tube_length - 1]:
            self.assertAlmostEqual(ftl_averages[i], 0)

    def test_FTL_constant_curvature_two_tubes(self):
        strain_bias = np.array([0, 0, 0, 1, 0, 0])
        strain_base = [constant_strain, constant_strain]

        model = Kinematic(2, [np.array([0.05]), np.array([0.03])], [1, 1], [50, 50], 0.5, strain_base)
        prev_g_out = model.solve_g([76, 78], [0, 0], full=False)
        g_out = model.solve_g([74, 77], [0, 0], full=False)

        _, ftl_out = model.solve_eta([-2, -1], [76, 78], [0, 0], prev_g_out, g_out)
        ftl_averages = [np.mean(array) for tube in ftl_out for array in tube]

        tube_length = len(ftl_out[-1])
        for i in [0, tube_length - 1]:
            self.assertAlmostEqual(ftl_averages[i], 0)
            self.assertNotEqual(ftl_averages[i + tube_length], 0)

    def test_Calculate_Indices_both_max(self):
        arr_neg = [-3, -1, 0]
        arr_max = [50, 51, 53]
        for i in range(3):
            for j in range(3):
                self.calc_indices_convenience(arr_neg[i], arr_neg[j], 0, 0, 0)
        for i in range(3):
            for j in range(3):
                self.calc_indices_convenience(arr_max[i], arr_max[j], 50, 50, 0)

    def test_Calculate_Indices_less_than_one(self):
        self.calc_indices_convenience(0, 0.49, 0, 1, 1)
        self.calc_indices_convenience(0.49, 0, 1, 0, -1)
        self.calc_indices_convenience(-5, 0.49, 0, 1, 1)
        self.calc_indices_convenience(0.49, -0.02, 1, 0, -1)
        self.calc_indices_convenience(49.51, 50, 49, 50, 1)
        self.calc_indices_convenience(50, 49.51, 50, 49, -1)
        self.calc_indices_convenience(49.4, 50.3, 49, 50, 1)
        self.calc_indices_convenience(60, 49.05, 50, 49, -1)
        self.calc_indices_convenience(10.2, 10.3, 10, 11, 1)
        self.calc_indices_convenience(10.3, 10.2, 11, 10, -1)

    def test_Calculate_Indices_round_prev_first(self):
        self.calc_indices_convenience(19.9, 21.95, 20, 22, 2)
        self.calc_indices_convenience(21.95, 19.9, 22, 20, -2)
        self.calc_indices_convenience(19.49, 21.51, 19, 21, 2)
        self.calc_indices_convenience(21.51, 19.49, 22, 20, -2)

    def test_Calculate_Indices_same_val(self):
        self.calc_indices_convenience(10, 10, 10, 10, 0)

    def test_Calculate_Indices_small_delta(self):
        self.calc_indices_convenience(10.27, 10.31, 103, 104, 1, delta_x=0.1)
        self.calc_indices_convenience(10.31, 10.27, 103, 102, -1, delta_x=0.1)

    def calc_indices_convenience(self, prev_init, next_init, prev_out, next_out, delta_out, max_tube=None, delta_x=1.):
        if max_tube is None:
            max_tube = [50]
        prev_insert_index, new_insert_index = calculate_indices([prev_init], [next_init], max_tube, delta_x)
        self.assertEqual(prev_insert_index, [prev_out])
        self.assertEqual(new_insert_index, [next_out])


if __name__ == '__main__':
    unittest.main()
