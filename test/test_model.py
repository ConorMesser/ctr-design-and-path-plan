import unittest
import numpy as np
import math

from ctrdapp.model import matrix_utils
from ctrdapp.model.model import Model
from ctrdapp.model.strain_bases import linear_strain


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

        self.no_strain1 = Model(1, no_strain_q1, 2, 1, 11, base1, strain_bias, 'Kinematic')
        self.no_strain1_rough = Model(1, no_strain_q1, 2, 1, 3, base1, strain_bias, 'Kinematic')
        self.no_strain1_fine = Model(1, no_strain_q1, 2, 1, 21, base1, strain_bias, 'Kinematic')
        self.no_strain2 = Model(2, no_strain_q2, 2, 1, 11, base2, strain_bias, 'Kinematic')
        self.constant_strain1 = Model(1, constant_q1, 2, math.pi/2, 11, base1, strain_bias, 'Kinematic')
        self.constant_strain1_fine = Model(1, constant_q1, 2, math.pi/2, 21, base1, strain_bias, 'Kinematic')
        self.constant_strain2 = Model(2, constant_q2, 2, math.pi/2, 11, base2, strain_bias, 'Kinematic')
        self.combo = Model(2, no_strain_1_constant_2, 2, 1, 11, base2, strain_bias, 'Kinematic')

        self.constant_tip = np.array([[0, 0, 1, 1],
                                      [0, 1, 0, 0],
                                      [-1, 0, 0, -1],
                                      [0, 0, 0, 1]])

    def test_solve_g_one_tube_no_strain(self):

        # Initial insertion
        no_strain1_g = self.no_strain1.solve_g()[0]
        no_strain1_g_disc = self.no_strain1_fine.solve_g()[0]
        eye_end = np.eye(4)
        eye_end[0, 3] = -1

        np.testing.assert_equal(no_strain1_g[-1], np.eye(4))
        np.testing.assert_almost_equal(no_strain1_g[0], eye_end)
        for i in range(11):
            np.testing.assert_almost_equal(no_strain1_g[i], no_strain1_g_disc[i*2])

        # Half insertion
        no_strain1_g = self.no_strain1.solve_g(indices=[5])[0]
        no_strain1_g_disc = self.no_strain1_fine.solve_g(indices=[10])[0]
        eye_end[0, 3] = -0.5

        np.testing.assert_equal(no_strain1_g[5], np.eye(4))
        np.testing.assert_almost_equal(no_strain1_g[0], eye_end)
        for i in range(11):
            np.testing.assert_almost_equal(no_strain1_g[i], no_strain1_g_disc[i * 2])

        # Full insertion
        no_strain1_g = self.no_strain1.solve_g(indices=[0])[0]
        no_strain1_g_disc = self.no_strain1_fine.solve_g(indices=[0])[0]
        eye_end[0, 3] = 1

        np.testing.assert_equal(no_strain1_g[0], np.eye(4))
        np.testing.assert_almost_equal(no_strain1_g[-1], eye_end)
        for i in range(11):
            np.testing.assert_almost_equal(no_strain1_g[i], no_strain1_g_disc[i * 2])

        # Theta tests
        no_strain1_g_rot_2pi = self.no_strain1.solve_g(indices=[0], thetas=[2*math.pi])[0]
        no_strain1_g_rot_halfpi = self.no_strain1.solve_g(thetas=[math.pi/2])[0]
        eye_90 = np.eye(4)
        eye_90[1:3, 1:3] = [[0, -1], [1, 0]]

        np.testing.assert_equal(no_strain1_g_rot_2pi[0], np.eye(4))
        np.testing.assert_almost_equal(no_strain1_g_rot_halfpi[-1], eye_90)
        for i in range(11):
            np.testing.assert_almost_equal(no_strain1_g[i], no_strain1_g_rot_2pi[i])

    def test_solve_g_one_tube_constant_strain(self):
        constant_strain1_g = self.constant_strain1.solve_g(indices=[0])[0]
        constant_strain1_g_fine = self.constant_strain1_fine.solve_g(indices=[0])[0]

        np.testing.assert_equal(constant_strain1_g[0], np.eye(4))
        np.testing.assert_almost_equal(constant_strain1_g[-1], self.constant_tip)
        for i in range(11):
            np.testing.assert_almost_equal(constant_strain1_g[i], constant_strain1_g_fine[i * 2])

    def test_solve_g_two_tubes_nostrain(self):
        no_strain2_g = self.no_strain2.solve_g()
        g_first_extended = self.no_strain2.solve_g(indices=[0, 10])

        for i in range(11):
            np.testing.assert_equal(no_strain2_g[0][i], no_strain2_g[1][i])
            np.testing.assert_almost_equal(g_first_extended[0][i], g_first_extended[1][i])

        no_strain2_g = self.no_strain2.solve_g(indices=[0, 0])
        np.testing.assert_equal(no_strain2_g[0][0], np.eye(4))
        np.testing.assert_equal(no_strain2_g[0][-1], no_strain2_g[1][0])

        end_g = np.eye(4)
        end_g[0, 3] = 2
        np.testing.assert_almost_equal(no_strain2_g[1][-1], end_g)

    def test_solve_g_two_tubes_constant(self):
        this_g = self.constant_strain2.solve_g(indices=[0, 0])
        turn_g = self.constant_strain2.solve_g(indices=[0, 0], thetas=[0, math.pi/2])

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
        a[0, 3] = -1
        b = np.eye(4)
        b[0, 3] = -0.5
        c = np.eye(4)
        d = np.eye(4)
        d[0, 3] = 0.5
        e = np.eye(4)
        e[0, 3] = 1
        g_3_no_strain = [[a, b, c]]

        # zero insertion
        g_test, eta_test, insert_indices, true_insertions, ftl_heuristic = self.no_strain1_rough.solve_integrate(
            [0], [0], [0], [0], invert_insert=True)
        np.testing.assert_equal(g_test, g_3_no_strain)
        np.testing.assert_equal(true_insertions, [0])
        np.testing.assert_equal(insert_indices, [2])

        # Round away from zero tests
        insert_half = [[b, c, d]]
        g_test_forward, eta_test, insert_indices, true_insertions, ftl_heuristic = self.no_strain1_rough.solve_integrate(
            [0], [0.499], [0], [0.499], invert_insert=True)
        np.testing.assert_equal(g_test_forward, insert_half)
        np.testing.assert_equal(true_insertions, [0.5])
        np.testing.assert_equal(insert_indices, [1])

        g_test_backward, eta_test, insert_indices, true_insertions, ftl_heuristic = self.no_strain1_rough.solve_integrate(
            [0], [-0.499], [0], [0.501], invert_insert=False)
        np.testing.assert_equal(g_test_backward, insert_half)
        np.testing.assert_equal(true_insertions, [0.5])
        np.testing.assert_equal(insert_indices, [1])

        insert_full = [[c, d, e]]
        g_test_forward, eta_test, insert_indices, true_insertions, ftl_heuristic = self.no_strain1_rough.solve_integrate(
            [0], [0.501], [0], [0.501], invert_insert=True)
        np.testing.assert_equal(g_test_forward, insert_full)
        np.testing.assert_equal(true_insertions, [1])
        np.testing.assert_equal(insert_indices, [0])

        g_test_backward, eta_test, insert_indices, true_insertions, ftl_heuristic = self.no_strain1_rough.solve_integrate(
            [0], [-0.501], [1], [0.499], invert_insert=False)
        np.testing.assert_equal(g_test_backward, insert_full)
        np.testing.assert_equal(true_insertions, [0])
        np.testing.assert_equal(insert_indices, [0])

        # Don't allow retraction past previous tube
        g_test, eta_test, insert_indices, true_insertions, ftl_heuristic = self.no_strain1_rough.solve_integrate(
            [0], [-0.1], [0], [-0.101], invert_insert=True)
        np.testing.assert_equal(g_test, g_3_no_strain)
        np.testing.assert_equal(true_insertions, [0])
        np.testing.assert_equal(insert_indices, [2])

        # Don't allow extension past max tube length
        g_test_forward, eta_test, insert_indices, true_insertions, ftl_heuristic = self.no_strain1_rough.solve_integrate(
            [0], [0.1], [0], [1.101], invert_insert=True)
        np.testing.assert_equal(g_test_forward, insert_full)
        np.testing.assert_equal(true_insertions, [1])
        np.testing.assert_equal(insert_indices, [0])

        # test for q_dot and eta

    def test_solve_once_speed(self):
        a = np.eye(4)
        a[0, 3] = -1
        b = np.eye(4)
        b[0, 3] = -0.5
        c = np.eye(4)
        g_3_no_strain = [[a, b, c]]

        # zero insertion
        for _ in range(5000):
            g_test_backward, eta_test, insert_indices, true_insertions = self.no_strain1_rough.solve_once(
                [0], [-0.501], [1], g_3_no_strain, invert_insert=False)

    def test_solve_g_speed(self):
        for _ in range(5000):
            turn_g = self.constant_strain2.solve_g(indices=[0, 0], thetas=[0, math.pi/2])
