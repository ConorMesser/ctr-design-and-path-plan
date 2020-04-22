import numpy as np
from math import ceil, floor
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

    def solve_integrate(self, delta_theta, delta_insertion, this_theta, this_insertion, prev_g, invert_insert=True):
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
        prev_g : list of list of 4x4 SE3 array
            SE3 g values for each tube from s to L of the previous insertion
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
        list : follow-the-leader array (g_dot minus g_prime) for each tube from s to L
            ftl_out[tube_num] = [6x1 array]
        """

        # kinematic model is defined as s(0)=L no insertion and s=0 for full insertion
        if invert_insert:
            this_insertion = [self.max_tube_length - ins for ins in this_insertion]
            delta_insertion = [-delta_ins for delta_ins in delta_insertion]

        # velocity is kept as given; it doesn't correspond to the discretization
        velocity = [-delta_ins for delta_ins in delta_insertion]
        prev_insertion = [ins - delta for ins, delta in zip(this_insertion, delta_insertion)]
        prev_insert_indices, new_insert_indices = calculate_indices(prev_insertion, this_insertion,
                                                                    self.max_tube_length, self.delta_x)

        # todo should I calculate prev_g???
        g_out = self.solve_g(indices=new_insert_indices, thetas=this_theta, full=False)
        eta_out, ftl_out = self.solve_eta(velocity, prev_insert_indices, delta_theta, prev_g)

        if invert_insert:
            true_insertions = [self.max_tube_length - (ind * self.delta_x) for ind in new_insert_indices]
        else:
            true_insertions = [ind * self.delta_x for ind in new_insert_indices]

        return g_out, eta_out, new_insert_indices, true_insertions, ftl_out

    def solve_g(self, indices=None, thetas=None, full=True):
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
        full : boolean (optional)
            Is the full tube needed (default) or just the segment past the insertion index?

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
        x_axis_unit = np.array([1, 0, 0, 0, 0, 0])

        for n in range(self.tube_num):
            theta_hat = hat(thetas[n] * x_axis_unit)
            theta_exp = exponential_map(thetas[n], theta_hat)
            g_previous = g_previous @ theta_exp
            this_g_tube = [g_previous]

            if full:
                index_start = 1
            else:
                index_start = indices[n] + 1

            # compute insertion w.r.t previous tube tip
            for i in range(index_start, self.num_discrete_points):  # todo Jacobian and q_dot
                centered_s = (i - 0.5) * self.delta_x
                this_centered_base = self.strain_base[n](centered_s, self.q_dof)

                ksi_here_center = this_centered_base @ self.q[n] + self.strain_bias
                norm_k_here_center = np.linalg.norm(ksi_here_center[0:3])

                gn_here = exponential_map(self.delta_x * norm_k_here_center,
                                          self.delta_x * hat(ksi_here_center))
                g_here = g_previous @ gn_here

                this_g_tube.append(g_here)
                g_previous = g_here

            # move tube to align index with previous tip (if full tube is desired)
            # if tube is fully extended (index == 0), already aligned
            if full and indices[n] != 0:
                # transform full tube to align insert index with previous tube tip
                transform = this_g_tube[0] @ np.linalg.inv(this_g_tube[indices[n]])  # g_initial X g_index^-1
                this_g_tube = [transform @ g for g in this_g_tube]
                g_previous = this_g_tube[-1]

            g_out.append(this_g_tube)

        return g_out

    def solve_eta(self, velocity_list, prev_insert_indices_list,
                  delta_theta_list, prev_g):
        eta_out = []
        ftl_out = []
        eta_previous_tube = np.zeros(6)  # in spatial frame
        x_axis_unit = np.array([1, 0, 0, 0, 0, 0])
        velocity_sum = 0

        for n in range(self.tube_num):
            # todo limit max delta_theta
            velocity_sum = velocity_sum + velocity_list[n]

            # strain_base must be list of functions, strain bias is np.array
            # q as list of n arrays of size dof
            ksi_here = self.strain_base[n](
                self.delta_x * prev_insert_indices_list[n], self.q_dof) @ self.q[n] + self.strain_bias

            # Adjoint of this tube's g_initial puts eta in spatial (world) frame
            eta_tr1 = big_adjoint(prev_g[n][0]) * delta_theta_list[n] @ x_axis_unit
            eta_tr2 = big_adjoint(prev_g[n][0]) * velocity_list[n] @ ksi_here

            if self.q_dot_bool:
                this_q_dot = None  # todo calculate q_dot - should be array with size q_dof
                jacobian_r_init = np.zeros([6, self.q_dof])
                eta_cr_here = jacobian_r_init @ this_q_dot
                jacobian_r_previous = jacobian_r_init
            else:
                eta_cr_here = 0

            eta_r_here = eta_previous_tube + eta_tr1 + eta_tr2 + eta_cr_here
            this_eta_r = [eta_r_here]
            this_ftl_heuristic = []

            for i in range(prev_insert_indices_list[n], self.num_discrete_points):
                this_prev_g_index = i - prev_insert_indices_list[n]
                if self.q_dot_bool and i != prev_insert_indices_list[n]:  # todo what about when tube inside other tube?
                    this_centered_base = self.strain_base[n]((i - 0.5) * self.delta_x, self.q_dof)
                    ksi_here_center = this_centered_base @ self.q[n] + self.strain_bias
                    norm_k_here_center = np.linalg.norm(ksi_here_center[0:3])

                    jacobian_r_here = jacobian_r_previous + big_adjoint(prev_g[n][this_prev_g_index-1]) @ t_exponential(
                        self.delta_x, norm_k_here_center, ksi_here_center) @ this_centered_base
                    eta_cr_here = jacobian_r_here @ this_q_dot
                    jacobian_r_previous = jacobian_r_here
                    eta_r_here = eta_previous_tube + eta_tr1 + eta_tr2 + eta_cr_here
                    this_eta_r.append(eta_r_here)

                # compute the g_prime based on the previous insertion
                this_base = self.strain_base[n](self.delta_x * i, self.q_dof)
                ksi_here = this_base @ self.q[n] + self.strain_bias
                g_prime = velocity_sum * big_adjoint(prev_g[n][this_prev_g_index]) @ ksi_here

                ftl_here = eta_r_here - g_prime
                this_ftl_heuristic.append(ftl_here)

            eta_previous_tube = eta_r_here  # big_adjoint(np.linalg.inv(g_previous)) @ eta_r_here todo check?
            eta_out.append(this_eta_r)  # eta is output w.r.t spatial frame
            ftl_out.append(this_ftl_heuristic)
        return eta_out, ftl_out

    def get_data(self):
        pass


def calculate_indices(prev_insertions, next_insertions, max_tube_length, delta_x):
    """Calculates indices from given insertion values with respect to certain rules.

    Neither the previous nor the next insertion values can go beyond the range [0, max_tube_length]. The final delta
    must be an integer and is equal to 0 only if prev_insertion = next_insertion (the delta_index is always rounded
    up to 1 when the inherent delta is <1 and rounded to the nearest integer in all other cases). The previous insertion
    is always rounded to the nearest multiple of delta_x first and the next_insertion value is calculated from it and
    the rounded delta. All outputs are integers corresponding to indices (with the size of index span given by delta_x).

    Parameters
    ----------
    prev_insertions : list of float
    next_insertions : list of float
    max_tube_length : float
    delta_x : float
        The discretization size (or size of one index)

    Returns
    -------
    list of int :
        the final index value for the previous insertion
    list of int :
        the final index value for the next insertion
    list of int :
        the delta between the two insertions (given as a number of indices)
    """
    prev_insert_indices = []
    new_insert_indices = []
    for n in range(len(prev_insertions)):
        prev_insertion = prev_insertions[n]
        next_insertion = next_insertions[n]

        if prev_insertion < 0:
            prev_insertion = 0
        elif prev_insertion > max_tube_length:
            prev_insertion = max_tube_length

        if next_insertion < 0:
            next_insertion = 0
        elif next_insertion > max_tube_length:
            next_insertion = max_tube_length

        delta_index = next_insertion - prev_insertion

        if abs(delta_index) <= (0.5 * delta_x):
            delta_index = np.sign(delta_index)
            if round(prev_insertion / delta_x) == round(next_insertion / delta_x) and (
                    ceil(prev_insertion / delta_x) == ceil(next_insertion / delta_x) or
                    floor(prev_insertion / delta_x) == floor(next_insertion / delta_x)):
                if delta_index < 0:
                    prev_insert_index = int(ceil(prev_insertion / delta_x))
                else:
                    prev_insert_index = int(floor(prev_insertion / delta_x))
            else:
                prev_insert_index = int(round(prev_insertion / delta_x))
        else:
            delta_index = round(delta_index / delta_x)
            prev_insert_index = int(round(prev_insertion / delta_x))

        new_insert_index = prev_insert_index + int(delta_index)
        prev_insert_indices.append(prev_insert_index)
        new_insert_indices.append(new_insert_index)

    return prev_insert_indices, new_insert_indices


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
