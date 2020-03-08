import numpy as np
from math import ceil, floor, copysign
import pathlib

from .matrix_utils import big_adjoint, hat, exponential_map, t_exponential
from .strain_bases import get_strains


class Model:

    def __init__(self, tube_num, q, q_dof, max_tube_length,
                 num_discrete_points, strain_base, strain_bias, model_type):
        self.tube_num = tube_num
        self.q_dof = q_dof
        self.max_tube_length = max_tube_length
        self.num_discrete_points = num_discrete_points
        self.strain_base = strain_base
        self.strain_bias = strain_bias
        self.q_dot_bool = model_type == 'static'

        if len(q) == tube_num * q_dof and not isinstance(q[0], list):  # q given as single list
            q_list = []
            for i in range(tube_num):
                q_list.append(q[i*q_dof:(i + 1)*q_dof])
            self.q = np.asarray(q_list)
        elif len(q) == tube_num:  # q given as nested-list
            self.q = np.asarray(q)
        else:
            raise ValueError(f"Given q of {q} is not the correct size.")

        self.delta_x = self.max_tube_length / (self.num_discrete_points - 1)

    def solve_iterate(self, delta_theta, delta_insertion,
                      previous_insertion, g_previous,
                      invert_insert=True):

        this_g_previous = g_previous
        this_previous_insertion = previous_insertion

        # get number of iterations needed (based on either delta, rounded up or based on prev + delta rounded??)
        # get max delta_ins = iter_num
        # get iter num for each tube
        # divide each delta_theta by the max iter_num

        # todo limit max delta_theta
        # todo check the limits for insertion/retraction

        iter_num = [ceil(abs(d_i / self.delta_x)) for d_i in delta_insertion]
        this_delta_theta = [d_t / max(iter_num) for d_t in delta_theta]  # integrate

        for i in range(max(iter_num)):

            this_delta_insertion = [copysign(self.delta_x, d_i) for d_i in delta_insertion]
            for n in range(self.tube_num):
                if i >= iter_num[n]:
                    this_delta_insertion[n] = 0

            g_out, eta_out, insert_indices, true_insertions = \
                self.solve_once(this_delta_theta, this_delta_insertion,
                                this_previous_insertion, this_g_previous,
                                invert_insert=invert_insert)
            this_g_previous = g_out
            this_previous_insertion = true_insertions

        return g_out, eta_out, insert_indices, true_insertions

    def solve_once(self, delta_theta, delta_insertion,
                   previous_insertion, g_previous,
                   invert_insert=True):
        """Calculate the g and eta for one step in time.

        Calculates the g dot based on the delta_theta and delta_insertion arrays
        and increments the given g_previous using the g_dot.

        Parameters
        ----------
        delta_theta : list-like of float
            change in theta values for each tube
        delta_insertion : list-like of float
            change in insertion values for each tube
        previous_insertion : list-like of float
            previous insertion values for each tube
        g_previous : list of list of array of float
            SE3 g values for the whole tube in the previous "time-step"
        invert_insert : bool
            True if insertion values are given intuitively (with s(0) = 0 and
            s(L) = L; false otherwise (default is True))

        Returns
        -------
        list : updated SE3 g values for each tube
            g_out[tube_num][index] = [4x4 SE3 array]
        list : eta value for each tube
            eta_out[tube_num][0] = eta -> (eta stored in list for consistency)
        """

        g_out = []
        eta_out = []
        insert_indices = []

        # kinematic model is defined as s(0)=L no insertion and s=0 for full insertion
        if invert_insert:
            previous_insertion = [self.max_tube_length - ins for ins in previous_insertion]
            delta_insertion = [-delta_ins for delta_ins in delta_insertion]

        eta_previous_tube = np.zeros(6)  # w.r.t tip of previous tube
        x_axis_unit = np.array([1, 0, 0, 0, 0, 0])

        for n in range(self.tube_num):
            velocity = -delta_insertion[n]

            # velocity should be rounded as multiple of delta_x
            if previous_insertion[n] < 0:  # greater than full insertion
                insert_index = 0
                if velocity > 0:
                    velocity = 0
            elif previous_insertion[n] < self.max_tube_length:
                insert_index = int(round(previous_insertion[n] / self.delta_x))
            else:  # retraction past tip of previous tube
                insert_index = self.num_discrete_points - 1
                if velocity < 0:
                    velocity = 0

            # round the new index value away from zero (small insertions move at least one index)
            delta_index = velocity / self.delta_x
            if delta_index > 0:
                new_insert_index = insert_index - ceil(delta_index)
            else:
                new_insert_index = insert_index - floor(delta_index)
            insert_indices.append(new_insert_index)
            velocity = (insert_index - new_insert_index) * self.delta_x  # todo ?????*****

            # strain_base must be list of numpy arrays, strain bias is np.array
            # q as list of n arrays of size dof
            ksi_here = self.strain_base[n](self.delta_x * insert_index, self.q_dof) @ self.q[n] + self.strain_bias
            omega = delta_theta[n] * x_axis_unit

            # relative velocity w.r.t. base of current tube
            eta_relative = velocity * ksi_here + omega

            # relative velocity w.r.t. fixed frame
            eta_relative_fixed = big_adjoint(g_previous[n][insert_index]) @ eta_relative

            # total velocity w.r.t fixed frame
            eta_total = eta_previous_tube + eta_relative_fixed

            this_q_dot = None
            if self.q_dot_bool:
                this_q_dot = None  # todo calculate q_dot - should be array with size q_dof

            norm_omega_here = np.linalg.norm(eta_total[0:3])

            # calculate new g from exponential of g_dot and previous g
            g_list = []
            eta_list = []
            jacobian_r_previous = np.zeros([6, self.q_dof])
            for i in range(len(g_previous[n])):

                p = g_previous[n][i]

                if self.q_dot_bool and i <= insert_index:  # before insertion, Jacobian = 0 (shape doesn't change)
                    centered_s = (i - 0.5) * self.delta_x
                    this_centered_base = self.strain_base[n](centered_s, self.q_dof)
                    ksi_here1 = this_centered_base @ self.q[n] + self.strain_bias
                    norm_k_here1 = np.linalg.norm(ksi_here1[0:3])

                    jacobian_r_here = jacobian_r_previous + big_adjoint(p) @ t_exponential(self.delta_x, norm_k_here1, ksi_here1) @ this_centered_base
                    eta_cr_here_relative_fixed = jacobian_r_here @ this_q_dot

                    eta_here = eta_total + eta_cr_here_relative_fixed
                    eta_list.append(eta_here)
                    this_g = exponential_map(norm_omega_here, hat(eta_here)) @ p
                else:
                    this_g = exponential_map(norm_omega_here, hat(eta_total)) @ p
                g_list.append(this_g)

            g_out.append(g_list)
            if self.q_dot_bool:
                eta_out.append(eta_list)
            else:
                eta_out.append([eta_total])

            eta_previous_tube = eta_out[n][-1]  # set eta_previous as sum of all tubes so far todo check this

        if invert_insert:
            true_insertions = [self.max_tube_length - (ind * self.delta_x) for ind in insert_indices]
        else:
            true_insertions = [ind * self.delta_x for ind in insert_indices]

        return g_out, eta_out, insert_indices, true_insertions

    def solve_g(self, indices=None, thetas=None):
        """Calculates the g of each point for each tube at given index and theta

        At default, the tips of each tube will be aligned with the origin, with
        no rotation or insertion. If indices are given, the given index of each
        tube will be aligned with the tip of the previous tube with theta
        rotation along x-axis wrt previous tube (*** check this ***)

        Parameters
        ---------
        indices : list of int (optional)
            Insertion index for each tube, with initial as default (no insertion)
        thetas : list of float (optional)
            Theta for each tube, with initial (0) as default

        Returns
        -------
        list : g values for each tube
            g_out[tube number] = [4x4 SE3 array]"""

        if indices is None:  # default to zero insertion, with s(0) = L
            indices = [self.num_discrete_points - 1] * self.tube_num
        if thetas is None:
            thetas = [0] * self.tube_num

        g_out = []
        g_previous = np.eye(4)
        g_previous_tip = np.eye(4)

        for n in range(self.tube_num):
            this_g = [g_previous]

            # compute full insertion w.r.t this tube base
            for i in range(1, self.num_discrete_points):
                centered_s = (i - 0.5) * self.delta_x
                this_centered_base = self.strain_base[n](centered_s, self.q_dof)

                ksi_here_center = this_centered_base @ self.q[n] + self.strain_bias
                norm_k_here_center = np.linalg.norm(ksi_here_center[0:3])

                gn_here = exponential_map(self.delta_x * norm_k_here_center,
                                          self.delta_x * hat(ksi_here_center))
                g_here = g_previous @ gn_here

                this_g.append(g_here)
                g_previous = g_here

            # transform full tube to align insert index with previous tube tip
            g_at_index = this_g[indices[n]]
            x_axis = np.array([1, 0, 0, 0, 0, 0])
            theta_exp = exponential_map(thetas[n],
                                        hat(thetas[n] * x_axis))
            this_g = [g_previous_tip @ (theta_exp @ np.linalg.inv(g_at_index) @ g)
                      for g in this_g]
            g_out.append(this_g)
            g_previous_tip = this_g[-1]
            g_previous = np.eye(4)

        return g_out

    def get_data(self):
        pass


def truncate_g(this_g, insert_indices):
    g_out = []
    for n, tube in enumerate(this_g):
        g_out.append(tube[insert_indices[n]:])

    return g_out


def save_g_positions(this_g, filename):
    path = pathlib.Path().absolute()
    file = path / "outputs" / filename

    f = open(file, "w")

    for i, tube in enumerate(this_g):
        f.write(f"Tube number: {i}\n\n")
        for g in tube:
            f.write(f"{g[0, 3]}, {g[1, 3]}, {g[2, 3]}\n")

    f.close()


def create_model(config, q):
    names = config.get('strain_bases').split(', ')
    strain_bases = get_strains(names)
    std_strain_bias = np.array([0, 0, 0, 1, 0, 0])

    return Model(config.get('tube_number'), q, config.get('q_dof'),
                 config.get('insertion_max'), config.get('num_discrete_points'),
                 strain_bases, std_strain_bias, config.get('model_type'))
