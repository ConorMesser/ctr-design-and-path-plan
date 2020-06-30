from math import sqrt, cos, sin, pi
import numpy as np
from scipy import interpolate, optimize

from .matrix_utils import big_adjoint, big_coadjoint, hat, exponential_map
from .kinematic import Kinematic


class Static(Kinematic):
    """The static strain-based model.

    Stores the configuration data for a model, and
    provides interface to solve for g based on
    given configuration (s, theta, v, omega, etc.).

    Parameters
    ----------
    tube_num : int
        number of tubes
    q : np.ndarray[list[float]] or list[np.ndarray[float]]
        bending parameters for each tube
        (will be multiplied by the strain bases)
    q_dof : list[int]
        number of degrees of freedom for each tube
    max_tube_length : list[float]
        maximum tube length for each tube
    delta_x : float
        number of discrete points along each tube
    strain_base : list[function]
        function defining each tube's strain base
    ndof : dict
        dictionary between (tube_num, section_num) and degrees of freedom
    radius : dict
        dictionary of 'inner' and 'outer' radii, {str : list[float]}

    Attributes
    ----------
    ndof : dict
        dictionary between (tube_num, section_num) and degrees of freedom
    q_big_order : list[tuple]
        order of the independent basis q where each tuple is (tube_number, section_number)
    stiffness_matrices : list[float]
        the calculated stiffness matrix based off the given radius
    gauss_coefficients_kinematic : list[float]
    gauss_coefficients_static : list[float]
    gauss_weights : list[float]
    section_indices : dict
        the given indices for each section, to be reset by solve_g
    theta : list[float]
        the given thetas, to be reset by solve_g
    """
    def __init__(self, tube_num, q, q_dof, max_tube_length, delta_x, strain_base, ndof, radius):
        super().__init__(tube_num, q, q_dof, max_tube_length, delta_x, strain_base)

        self.ndof = ndof
        self.q_big_order = [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 3)]  # how to use?
        self.stiffness_matrices = calculate_stiffness(radius)  # where to calculate?
        self.gauss_coefficients_kinematic, self.gauss_coefficients_static, self.gauss_weights = calculate_gauss()
        self.section_indices = None
        self.theta = None

    def solve_g(self, indices=None, thetas=None, full=True):  # todo I don't want to do full
        if indices is None:  # default to zero insertion, with s(0) = L
            indices = [disc - 1 for disc in self.num_discrete_points]
        if thetas is None:
            thetas = [0] * self.tube_num

        # calculate ini's
        index_11 = indices[0]
        index_22 = indices[1]
        index_33 = indices[2]
        index_21 = index_22 - (self.num_discrete_points[0] - index_11)
        index_32 = index_33 - (self.num_discrete_points[1] - index_22)
        index_31 = index_32 - (self.num_discrete_points[0] - index_11)

        section_indices = {(1, 1): index_11, (2, 1): index_21, (3, 1): index_31,
                           (2, 2): index_22, (3, 2): index_32, (3, 3): index_33}
        self.section_indices = section_indices
        self.theta = thetas  # type: list

        initial_guess = np.asarray([0, 0, 0, 0, 0, 0])  # todo get from previous run
        solution = optimize.root(self.static_equilibrium, initial_guess, method='lm')

        # first section
        g_11 = self.integrate_g(solution, eye_from_theta(self.theta[0]), 1, 1)
        g_21 = self.integrate_g(solution, eye_from_theta(self.theta[1]), 2, 1)
        g_31 = self.integrate_g(solution, eye_from_theta(self.theta[2]), 3, 1)

        # second section
        g_22 = self.integrate_g(solution, g_11[-1] @ g_21[-1], 2, 2)  # todo why is it g_11 * g_21
        g_32 = self.integrate_g(solution, np.eye(4), 3, 2)  # why is it eye(4)???

        # third section
        g_33 = self.integrate_g(solution, g_22[-1] @ g_31[-1] @ g_32[-1], 3, 3)  # todo why???

        g_full = [g_11, g_21 + g_22, g_31 + g_32 + g_33]

        self.section_indices = None
        self.theta = None
        return g_full

    def integrate_g(self, solution, prev_g, tube_num, section_num):
        q = self.select_from_array(solution, self.q_big_order, tube_num, section_num)
        this_g = prev_g
        collect_g = [this_g]

        index_length_section = self.num_discrete_points[section_num] - self.section_indices.get(
            (tube_num, section_num)) + 1
        for i in range(1, index_length_section):
            index_adjusted = i + self.section_indices.get((tube_num, section_num))
            _, this_g = self.half_step_integral(index_adjusted + self.gauss_coefficients_kinematic[0],
                                                index_adjusted + self.gauss_coefficients_kinematic[1], 0, q, this_g,
                                                tube_num, section_num)
            collect_g.append(this_g)

        return collect_g

    def select_from_array(self, arr, arr_order, tube_num, section_num):

        end_ind = arr_order.index((tube_num, section_num))
        flatten_dof = [self.ndof.get(item) for item in arr_order]
        if end_ind == 0:
            return arr[0:flatten_dof[0]]
        else:
            return arr[sum(flatten_dof[0:end_ind]):sum(flatten_dof[0:end_ind + 1])]

    def static_equilibrium(self, q):
        """Provides the static equilibrium q based off the given q???

        Parameters
        ----------
        q : list[float]

        Returns
        -------
        list[float]
        """

        # ---------------- first section --------------
        q_1 = [q[0:self.ndof[0, 0]],
               q[self.ndof[0, 0]:self.ndof[0, 0] + self.ndof[1, 0]],
               q[self.ndof[0, 0] + self.ndof[1, 0]:
                 self.ndof[0, 0] + self.ndof[1, 0] + self.ndof[2, 0]]]

        ar_intc1 = []
        ar_intg21 = []
        ar_intg31 = []

        sg21 = np.zeros(6, self.ndof[1, 0])
        sg31 = np.zeros(6, self.ndof[2, 0])
        gg21 = eye_from_theta(self.theta[1])
        gg31 = eye_from_theta(self.theta[2])

        # iterate over length
        index_length_section_1 = self.num_discrete_points[0] - self.section_indices.get((0, 0)) + 1
        for i in range(index_length_section_1):
            index_adjusted = [i + self.section_indices.get((j, 0)) for j in range(3)]

            bc11_here = get_basis(index_adjusted[0], self.delta_x, 1, 1, self.strain_base, self.q_dof)
            bg21_here = get_basis(index_adjusted[1], self.delta_x, 2, 1, self.strain_base, self.q_dof)
            bg31_here = get_basis(index_adjusted[2], self.delta_x, 3, 1, self.strain_base, self.q_dof)

            ksi_1_here = bc11_here @ q_1[0] + self.strain_bias
            ksi_21_here = bg21_here @ q_1[1]
            ksi_31_here = bg31_here @ q_1[2]
            coadj_ksi_1_here = big_coadjoint(ksi_1_here)

            if i != 0:
                sg21, gg21 = self.half_step_integral(index_adjusted[1] + self.gauss_coefficients_kinematic[0],
                                                     index_adjusted[1] + self.gauss_coefficients_kinematic[1], sg21,
                                                     q_1[1], gg21, 2, 1)
                sg31, gg31 = self.half_step_integral(index_adjusted[2] + self.gauss_coefficients_kinematic[0],
                                                     index_adjusted[2] + self.gauss_coefficients_kinematic[1], sg31,
                                                     q_1[2], gg31, 3, 1)

            adj_g21_here = big_adjoint(gg21)
            adj_g31_here = big_adjoint(gg31)
            coadj_g21_here = big_coadjoint(gg21)
            coadj_g31_here = big_coadjoint(gg31)

            # diff
            ksi_21_here = np.linalg.inv(adj_g21_here) @ ksi_1_here + ksi_21_here
            ksi_31_here = np.linalg.inv(adj_g31_here) @ np.linalg.inv(
                adj_g21_here) @ ksi_1_here + np.linalg.inv(adj_g31_here) @ ksi_21_here + ksi_31_here

            fi_11_here = self.stiffness_matrices[0] @ (
                    ksi_1_here - get_ksi_star(index_adjusted[0], self.delta_x, 1, self.strain_base[0], self.q[0],
                                              self.strain_bias))
            fi_21_here = self.stiffness_matrices[1] @ (
                    ksi_21_here - get_ksi_star(index_adjusted[1], self.delta_x, 2, self.strain_base[1], self.q[1],
                                               self.strain_bias))
            fi_31_here = self.stiffness_matrices[2] @ (
                    ksi_31_here - get_ksi_star(index_adjusted[2], self.delta_x, 3, self.strain_base[2], self.q[2],
                                               self.strain_bias))

            # diff
            ar_intc1.append(bc11_here.transpose() @ (
                    fi_11_here + coadj_g21_here @ fi_21_here +
                    coadj_g21_here @ coadj_g31_here @ fi_31_here))
            ar_intg21.append(bg21_here.transpose() @ (fi_21_here + fi_31_here) -
                             sg21.transpose() @ coadj_ksi_1_here @ coadj_g21_here @ (
                                     fi_21_here + coadj_g31_here @ fi_31_here))
            ar_intg31.append(bg31_here.transpose() @ fi_31_here -
                             sg31.transpose() @ coadj_ksi_1_here @
                             coadj_g21_here @ coadj_g31_here @ fi_31_here)

        # calculate integral
        intc_1 = self.calculate_integral(ar_intc1, index_length_section_1, self.ndof[0, 0],
                                         self.gauss_coefficients_static, self.gauss_weights)
        intg_21 = self.calculate_integral(ar_intg21, index_length_section_1, self.ndof[1, 0],
                                          self.gauss_coefficients_static, self.gauss_weights)
        intg_31 = self.calculate_integral(ar_intg31, index_length_section_1, self.ndof[2, 0],
                                          self.gauss_coefficients_static, self.gauss_weights)

        # -------------------- Second Section --------------
        # arranged as: [q_2, q_32]
        q_2 = [q[sum(self.ndof[0:3, 0]):
                 sum(self.ndof[0:3, 0]) + self.ndof[1, 1]],
               q[sum(self.ndof[0:3, 0]) + self.ndof[1, 1]:
                 sum(self.ndof[0:3, 0]) + sum(self.ndof[1:3, 1])]]

        # need 2, 32, 312
        ar_intc2 = []
        ar_intg32 = []
        ar_intg312 = []

        # need 32: zeros, 312: 31_previous
        sg32 = np.zeros(6, self.ndof[2, 1])
        sg312 = sg31

        # need 32: eye, 31: 31_previous
        gg32 = np.eye(4)
        # gg31 = previous gg31
        adj_g31_here = big_adjoint(gg31)
        coadj_g31_here = big_coadjoint(gg31)

        # iterate over length
        index_length_section_2 = self.num_discrete_points[1] - self.section_indices.get((1, 1)) + 1
        for i in range(index_length_section_2):
            index_adjusted = [i + self.section_indices.get((j, 1)) for j in range(1, 3)]

            # need 2, 32
            bc22_here = get_basis(index_adjusted[0], self.delta_x, 2, 2, self.strain_base, self.q_dof)
            bg32_here = get_basis(index_adjusted[1], self.delta_x, 3, 2, self.strain_base, self.q_dof)

            # need 2, 32
            ksi_2_here = bc22_here @ q_2[0] + self.strain_bias
            ksi_32_here = bg32_here @ q_2[1]
            coadj_ksi_2_here = big_coadjoint(ksi_2_here)

            if i != 0:
                # need only 32
                sg32, gg32 = self.half_step_integral(index_adjusted[1] + self.gauss_coefficients_kinematic[0],
                                                     index_adjusted[1] + self.gauss_coefficients_kinematic[1], sg32,
                                                     q_2[1], gg32, 3, 2)

            # only 32
            adj_g32_here = big_adjoint(gg32)
            coadj_g32_here = big_coadjoint(gg32)

            # only 32: add in a np.linalg.inv(adgg31) after adgg32
            ksi_32_here = np.linalg.inv(adj_g32_here) @ np.linalg.inv(adj_g31_here) @ ksi_2_here + ksi_32_here

            fi_22_here = self.stiffness_matrices[1] @ (ksi_2_here - get_ksi_star(index_adjusted[0], self.delta_x, 2,
                                                                                 self.strain_base[1], self.q[1],
                                                                                 self.strain_bias))
            fi_32_here = self.stiffness_matrices[2] @ (ksi_32_here - get_ksi_star(index_adjusted[1], self.delta_x, 3,
                                                                                  self.strain_base[2], self.q[2],
                                                                                  self.strain_bias))

            # different
            ar_intc2.append(bc22_here.transpose() @ (
                    fi_22_here + coadj_g31_here @ coadj_g32_here @ fi_32_here))
            ar_intg32.append(bg32_here.transpose() @ fi_32_here -
                             sg32.transpose() @ coadj_ksi_2_here @
                             coadj_g31_here @ coadj_g32_here @ fi_32_here)
            ar_intg312.append(sg312.transpose() @ coadj_ksi_2_here @
                              coadj_g31_here @ coadj_g32_here @ fi_32_here)

        # calculate integral!
        # just x_vals and length different (and input integral values)
        intc_2 = self.calculate_integral(ar_intc2, index_length_section_2, self.ndof[1, 1],
                                         self.gauss_coefficients_static, self.gauss_weights)
        intg_32 = self.calculate_integral(ar_intg32, index_length_section_2, self.ndof[2, 1],
                                          self.gauss_coefficients_static, self.gauss_weights)
        intg_312 = self.calculate_integral(ar_intg312, index_length_section_2, self.ndof[2, 0],
                                           self.gauss_coefficients_static, self.gauss_weights)

        # ---------------------- Section Three --------------------
        q_3 = q[sum(self.ndof[0:3, 0]) + sum(self.ndof[1:3, 1]):
                sum(self.ndof[0:3, 0]) + sum(self.ndof[1:3, 1]) + self.ndof[2, 2]]

        ar_intc3 = []
        index_length_section_3 = self.num_discrete_points[2] - self.section_indices.get((2, 2)) + 1
        for i in range(index_length_section_3):
            index_adjusted = i + self.section_indices.get((2, 2))

            bc3_here = get_basis(index_adjusted, self.delta_x, 3, 3, self.strain_base, self.q_dof)

            ksi_3_here = bc3_here @ q_3 + self.strain_bias
            fi_33_here = self.stiffness_matrices[2] @ (
                    ksi_3_here - get_ksi_star(index_adjusted, self.delta_x, 3, self.strain_base[2], self.q[2],
                                              self.strain_bias))
            ar_intc3.append(bc3_here.transpose() @ fi_33_here)
        intc_3 = self.calculate_integral(ar_intc3, index_length_section_3, self.ndof[2, 2],
                                         self.gauss_coefficients_static, self.gauss_weights)

        return np.concatenate((intc_1, intg_21, intg_31 - intg_312, intc_2, intg_32, intc_3))

    def calculate_integral(self, ar_int, num_points, ndof, constants, weights):  # todo test
        length = (num_points - 1) * self.delta_x
        x_vals = np.linspace(0, length, num_points)
        integral_collect = np.zeros([5, ndof])
        for i in range(ndof):
            interp_function = interpolate.interp1d(x_vals, ar_int[i, :])
            this_iteration = [interp_function(val * length) for val in constants]
            integral_collect[:, i] = this_iteration

        combine = [w * integ for w, integ in zip(weights, integral_collect)]
        combine2 = np.asarray(combine)
        iteration_out = length / 2 * combine2.sum(0)
        return iteration_out

    def half_step_integral(self, step1, step2, sg, this_q, gg, tube_number, section_number):
        bg_here1 = get_basis(step1, self.delta_x, tube_number, section_number, self.strain_base, self.q_dof)
        bg_here2 = get_basis(step2, self.delta_x, tube_number, section_number, self.strain_base, self.q_dof)
        sg = sg + (self.delta_x / 2) * (bg_here1 + bg_here2)

        if tube_number == section_number:
            ksig_hat_here1 = hat(bg_here1 @ this_q + self.strain_bias)
            ksig_hat_here2 = hat(bg_here2 @ this_q + self.strain_bias)
        else:
            ksig_hat_here1 = hat(bg_here1 @ this_q)
            ksig_hat_here2 = hat(bg_here2 @ this_q)

        # todo overkill - just an angle
        gammag_hat_here = (self.delta_x / 2) * (ksig_hat_here1 + ksig_hat_here2) + (
                (sqrt(3) * self.delta_x ** 2) / 12) * (
                                  ksig_hat_here2 @ ksig_hat_here1 - ksig_hat_here1 @ ksig_hat_here2)
        kg_here = np.transpose([-gammag_hat_here[1, 2], gammag_hat_here[0, 2], -gammag_hat_here[0, 1]])

        ggn_here = exponential_map(np.linalg.norm(kg_here), gammag_hat_here)
        gg = gg @ ggn_here

        return sg, gg


def get_ksi_star(index, dx, this_tube_num, strain_base, q, strain_bias):
    base_here = strain_base(index, dx, this_tube_num) @ q + strain_bias
    return base_here @ q + strain_bias


def get_basis(index, dx, this_tube_num, tube_section, strain_bases, dof):
    """
    If 33, just give Base from 3 (with index? or other?)
    All others get basis with one, X, X^2 (index should be fine)
    For remaining two or three bases, gives B1/B2/B3 with some transformations
        Careful about indices here - need to investigate it in MATLAB

    What to do about different tube numbers?

    todo would I rather save time or memory (function or storage) - can be solved in black box

    Parameters
    ----------
    index : float
    dx : float
    this_tube_num : int
    tube_section : int
    strain_base : list[function]
    dof : list[int]

    Returns
    -------
    np.ndarray
    """
    x_val = index * dx
    this_strain_base = strain_bases[this_tube_num - 1]
    if this_tube_num == tube_section:
        if this_tube_num == 3:  # todo or change to 2?
            base = this_strain_base(x_val, dof[2])
            pass
        else:
            base = np.zeros((6, 9))
            base[0, 0] = base[1, 3] = base[2, 6] = 1
            base[0, 1] = base[1, 4] = base[2, 7] = x_val
            base[0, 2] = base[1, 5] = base[2, 8] = x_val ** 2
    else:
        base = np.zeros((6, 3))
        base[0, 0] = 1
        base[0, 1] = x_val
        base[0, 2] = x_val ** 2

    return base


def calculate_stiffness(radii):
    r_out = radii.get('outer')  # todo
    r_in = radii.get('inner')
    area = [pi * (o ** 2 - i ** 2) for o, i in zip(r_out, r_in)]
    # Jy = Jz
    polar_mom = [pi / 4 * (o ** 4 - i ** 4) for o, i in zip(r_out, r_in)]
    mom_inertia = polar_mom * 2
    e = 60e9
    g = 23.1e9

    return [np.diag(np.asarray([g * i, e * j, e * j, e * a, g * a, g * a])) for
            i, j, a in zip(mom_inertia, polar_mom, area)]


def calculate_gauss():  # todo use fewer order quadrature?
    cg1 = 1 / 2 - sqrt(3) / 6  # Gauss quadrature coefficient for kinematics
    cg2 = 1 / 2 + sqrt(3) / 6
    kinematic = [cg1, cg2]
    c1 = 1 / 2 - (1 / 3) * sqrt(5 - 2 * sqrt(10 / 7)) / 2  # Gauss quadrature coefficients for statics
    c2 = 1 / 2 + (1 / 3) * sqrt(5 - 2 * sqrt(10 / 7)) / 2
    c3 = 1 / 2 - (1 / 3) * sqrt(5 + 2 * sqrt(10 / 7)) / 2
    c4 = 1 / 2 + (1 / 3) * sqrt(5 + 2 * sqrt(10 / 7)) / 2
    static = [c1, c2, c3, c4, 0.5]
    # Gauss quadrature weights
    w1 = (322 + 13 * sqrt(70)) / 900
    w3 = (322 - 13 * sqrt(70)) / 900
    w5 = 128 / 225
    weights = [w1, w1, w3, w3, w5]
    return kinematic, static, weights


def eye_from_theta(theta):
    return np.asarray([[1, 0, 0, 0],
                       [0, cos(theta), -sin(theta), 0],
                       [0, sin(theta), cos(theta), 0],
                       [0, 0, 0, 1]])
