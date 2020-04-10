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
        self.q_dot_bool = model_type == 'Static'

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

    def solve_integrate(self, delta_theta, delta_insertion,
                        this_theta, this_insertion, invert_insert=True):
        """Calculate the g and eta for one step in space.

        Calculates the g_prime based on the theta and insertion arrays
        and reconstructs this g based on the g_prime. The eta is calculated
        using the theta and insertion deltas as well as the calculated q_dot.

        Parameters
        ----------
        delta_theta : list-like of float
            change in theta values for each tube
        delta_insertion : list-like of float
            change in insertion values for each tube
        this_theta : list-like of float
            rotation values for each tube
        this_insertion : list-like of float
            insertion values for each tube
        invert_insert : bool
            True if insertion values are given intuitively (with s(0) = 0 and
            s(L) = L; false otherwise (default is True))

        Returns
        -------
        list : updated SE3 g values for each tube from s to L
            g_out[tube_num][index] = [4x4 SE3 array]
        list : eta value for each tube
            eta_out[tube_num][0] = eta -> (eta stored in list for consistency)
        list : insertion indices for each tube
            insert_indices[tube_num] = int
        list : true insertion values (rounded based on discretization)
            true_insertions[tube_num] = float
        list : follow-the-leader array (g_dot minux g_prime) for each tube from s to L
            ftl_out[tube_num] = [6x1 array]
        """
        g_out = []
        ksi_out = []
        eta_out = []
        ftl_out = []
        insert_indices = []

        velocity_sum = 0

        # kinematic model is defined as s(0)=L no insertion and s=0 for full insertion
        if invert_insert:
            this_insertion = [self.max_tube_length - ins for ins in this_insertion]
            delta_insertion = [-delta_ins for delta_ins in delta_insertion]

        g_previous = np.eye(4)
        eta_previous_tube = np.zeros(6)  # w.r.t tip of previous tube
        x_axis_unit = np.array([1, 0, 0, 0, 0, 0])

        for n in range(self.tube_num):
            # todo limit max delta_theta
            # todo check the limits for insertion/retraction
            # todo what is the velocity? Do we want to stop an insertion that already inserted too far?
            # is the delta_insertion from the previous point to this point? If it is illegitimate
            # what happens? You take the negative insertion velocity from this_insertion?
            velocity = -delta_insertion[n]

            # this_insertion value is limited by 0 and L
            # velocity is adjusted to correspond with the updated insertion value
            if this_insertion[n] < 0:  # greater than full insertion
                insert_index = 0
                velocity = velocity + this_insertion[n]
            elif this_insertion[n] < self.max_tube_length:  # todo round up if velocity less than delta_x?
                insert_index = int(round(this_insertion[n] / self.delta_x))
                velocity = velocity - (insert_index * self.delta_x - this_insertion[n])
            else:  # retraction past tip of previous tube
                insert_index = self.num_discrete_points - 1
                velocity = velocity + this_insertion[n] - self.max_tube_length  # todo

            velocity_sum = velocity_sum + velocity

            insert_indices.append(insert_index)

            # strain_base must be list of functions, strain bias is np.array
            # q as list of n arrays of size dof
            ksi_here = self.strain_base[n](self.delta_x * insert_index, self.q_dof) @ self.q[n] + self.strain_bias

            #  todo ksi_here should be based on previous insert/theta???

            theta_hat = hat(this_theta[n] * x_axis_unit)
            g_theta = exponential_map(this_theta[n], theta_hat)
            g_initial = g_previous @ g_theta

            eta_tr1 = big_adjoint(g_initial) * delta_theta[n] @ x_axis_unit  # should this be g_init or g_theta?? todo
            eta_tr2 = big_adjoint(g_initial) * velocity @ ksi_here

            if self.q_dot_bool:
                this_q_dot = None  # todo calculate q_dot - should be array with size q_dof
                jacobian_r_init = np.zeros([6, self.q_dof])
                eta_cr_here = jacobian_r_init @ this_q_dot
                jacobian_r_previous = jacobian_r_init
            else:
                eta_cr_here = 0

            eta_r_here = eta_previous_tube + eta_tr1 + eta_tr2 + eta_cr_here

            this_g = [g_initial]
            this_ksi_c = [ksi_here]
            this_eta_r = [eta_r_here]
            this_ftl_heuristic = []  # initial todo

            g_previous = g_initial

            for i in range(insert_index + 1, self.num_discrete_points):
                this_base = self.strain_base[n](self.delta_x * i, self.q_dof)
                centered_s = (i - 0.5) * self.delta_x
                this_centered_base = self.strain_base[n](centered_s, self.q_dof)

                ksi_here = this_base @ self.q[n] + self.strain_bias
                ksi_here_center = this_centered_base @ self.q[n] + self.strain_bias
                norm_k_here_center = np.linalg.norm(ksi_here_center[0:3])

                gn_here = exponential_map(self.delta_x * norm_k_here_center,
                                          self.delta_x * hat(ksi_here_center))
                g_here = g_previous @ gn_here

                if self.q_dot_bool:  # todo what about when tube is inside other tube?
                    jacobian_r_here = jacobian_r_previous + big_adjoint(g_previous) @ t_exponential(self.delta_x, norm_k_here_center, ksi_here_center) @ this_centered_base
                    eta_cr_here = jacobian_r_here @ this_q_dot
                    jacobian_r_previous = jacobian_r_here
                    eta_r_here = eta_previous_tube + eta_tr1 + eta_tr2 + eta_cr_here
                    this_eta_r.append(eta_r_here)

                # update for multi-tube (velocity, adjoint) todo
                g_prime = velocity_sum * big_adjoint(g_here) @ ksi_here

                ftl_here = eta_r_here - g_prime

                this_g.append(g_here)
                this_ksi_c.append(ksi_here)
                this_ftl_heuristic.append(ftl_here)

                g_previous = g_here
            eta_previous_tube = eta_r_here  # big_adjoint(np.linalg.inv(g_previous)) @ eta_r_here
            g_out.append(this_g)
            eta_out.append(this_eta_r)
            ksi_out.append(this_ksi_c)
            ftl_out.append(this_ftl_heuristic)

        if invert_insert:
            true_insertions = [self.max_tube_length - (ind * self.delta_x) for ind in insert_indices]
        else:
            true_insertions = [ind * self.delta_x for ind in insert_indices]

        return g_out, eta_out, insert_indices, true_insertions, ftl_out

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
