def _solve_once_integrate_space(self, delta_theta, delta_insertion,
                                this_theta, this_insertion,
                                invert_insert=True):
    delta_x = self.max_tube_length / (self.num_discrete_points - 1)
    g_out = []
    ksi_out = []
    eta_out = []

    if invert_insert:
        this_insertion = [ins - self.max_tube_length for ins in this_insertion]
        delta_insertion = [-delta_ins for delta_ins in delta_insertion]

    g_previous = np.eye(4)
    eta_r_previous = np.zeros(6)  # w.r.t tip of previous tube
    x_axis_unit = np.array([1, 0, 0, 0, 0, 0])

    for n in range(self.tube_num):
        if this_insertion[n] < self.max_tube_length:
            insert_index = round(this_insertion[n] / delta_x)
        else:  # doesn't allow for retraction past tip of previous tube
            insert_index = self.num_discrete_points - 1

        this_g = []
        this_ksi_c = []
        this_eta_r = []

        omega_x = delta_theta[n]
        velocity = -delta_insertion[n]

        if (velocity > 0 >= this_insertion[n]) or \
                (velocity < 0 <= this_insertion[n]):
            velocity = 0

        # for q with (n rows and dof columns)
        ksi_o = self.strain_base[insert_index] @ self.q[n, :] + self.strain_bias

        theta_hat = dynamic_hat(this_theta[n] * x_axis_unit)
        g_theta = variable_exponential_map(this_theta[n], theta_hat)
        g_initial = g_previous @ g_theta

        eta_tr1 = dynamic_adjoint(g_theta) * omega_x @ x_axis_unit
        eta_tr2 = dynamic_adjoint(g_theta) * velocity @ ksi_o


        q_dot_h_now = self.q_dot[n, :]
        jacobian_r_init = np.zeros(6, self.q_dof)
        eta_cr_o = jacobian_r_init @ q_dot_h_now

        eta_r_here = eta_r_previous + eta_tr1 + eta_tr2 + eta_cr_o

        this_g.append(g_initial)
        this_ksi_c.append(ksi_o)
        this_eta_r.append(eta_r_here)

        lin_x = range(0, self.num_discrete_points)
        g_previous = g_initial
        jacobian_r_previous = jacobian_r_init
        for i in range(insert_index + 1, self.num_discrete_points):
            this_base = self.strain_base[i]
            this_base1 = np.zeros([6, self.q_dof])

            for d in range(6):
                for q in range(self.q_dof):
                    this_base1[d, q] = np.interp(i - 0.5, lin_x, [b[d, q] for b in self.strain_base])  # check todo

            ksi_here = this_base @ self.q[n, :] + self.strain_bias
            ksi_here1 = this_base1 @ self.q[n, :] + self.strain_bias
            ksi_hat_here1 = dynamic_hat(ksi_here1)
            norm_k_here1 = np.linalg.norm(ksi_here1[0:3])

            gn_here = variable_exponential_map(delta_x * norm_k_here1,
                                               delta_x * ksi_hat_here1)
            g_here = g_previous @ gn_here

            jacobian_r_here = jacobian_r_previous + dynamic_adjoint(g_previous) @ variable_t_exponential(delta_x,
                                                                                                         norm_k_here1,
                                                                                                         ksi_here1) @ this_base1
            eta_cr_here = jacobian_r_here @ q_dot_h_now
            eta_r_here = eta_r_previous + eta_tr1 + eta_tr2 + eta_cr_here

            # duplicate code!!!!*****
            this_g.append(g_here)
            this_ksi_c.append(ksi_here)
            this_eta_r.append(eta_r_here)

            g_previous = g_here
            jacobian_r_previous = jacobian_r_here
        eta_r_previous = dynamic_adjoint(np.linalg.inv(g_previous)) @ eta_r_here
        g_out.append(this_g)
        eta_out.append(this_eta_r)
        ksi_out.append(this_ksi_c)