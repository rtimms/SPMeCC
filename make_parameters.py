class Parameters(object):
    def __init__(self, C_rate=1, name="mypouch"):
        """
        The parameters for the simulation.
        Parameters
        ----------
        C_rate : float
                 The C-rate for the discharge.
        name : string
                The name of the dimensional paramters to load.
        """
        # Put C-rate in Parameters object
        self.C_rate = C_rate

        # Put name in parameters object
        self.name = name

        # Load up dimensional parameters
        if self.name == "mypouch" or "mypouch_B":

            # Geometry
            self.L_cn_star = 25.0 * 1e-6
            self.L_n_star = 100.0 * 1e-6
            self.L_s_star = 25.0 * 1e-6
            self.L_p_star = 100.0 * 1e-6
            self.L_cp_star = 25 * 1e-6

            if self.name == "mypouch":
                self.L_tab_n_star = 48 * 1e-3
                self.L_tab_p_star = 48 * 1e-3
                self.tab_offset_star = 10 * 1e-3
                self.tab_n_location = "t"  # Location of negative tab (t, b, l, r)
                self.tab_p_location = "t"  # Location of negative tab (t, b, l, r)
            elif self.name == "mypouch_B":
                self.L_tab_n_star = 48 * 1e-3
                self.L_tab_p_star = 48 * 1e-3
                self.tab_offset_star = 66 * 1e-3
                self.tab_n_location = "t"  # Location of negative tab (t, b, l, r)
                self.tab_p_location = "b"  # Location of negative tab (t, b, l, r)

            self.A_tab_n_star = self.L_cn_star * self.L_tab_n_star
            self.A_tab_p_star = self.L_cp_star * self.L_tab_n_star

            self.Lx_star = self.L_n_star + self.L_s_star + self.L_p_star
            self.Ly_star = 180 * 1e-3
            self.Lz_star = 220 * 1e-3

            self.L_star = self.L_cn_star + self.Lx_star + self.L_cp_star

            # Porosity
            # self.epsilon_n = 0.485
            # self.epsilon_s = 0.724
            # self.epsilon_p = 0.385
            # self.brug = 4

            # Porosity
            self.epsilon_n = 0.3
            self.epsilon_s = 1.0
            self.epsilon_p = 0.3
            self.brug = 1.5

            # Filler fractions
            self.epsilon_f_n = 0.0326
            self.epsilon_f_p = 0.025

            # Cutoff voltage
            self.V_min = 2.5

            # Cutover voltage
            self.V_max = 4.3

            # Applied current density
            self.I_app_1C = 1.0
            self.I_star = self.C_rate * self.I_app_1C / self.Lz_star ** 2

            # Electrical conductivity
            self.sigma_cn_star = 1 * 1e7
            self.sigma_n_star = 100
            self.sigma_p_star = 100
            self.sigma_cp_star = 0.5 * 1e7

            # Diffusivity
            self.D_n_tilde_star = 3.9 * 1e-14
            self.D_p_tilde_star = 1 * 1e-14
            self.D_e_typ_star = 5.34 * 1e-10

            # Particle
            self.c_n_max_star = 24983
            self.c_p_max_star = 51218
            self.R_n_star = 10 * 1e-6
            self.R_p_star = 10 * 1e-6
            self.a_n_star = 0.18 * 1e6
            self.a_p_star = 0.15 * 1e6

            # Electrochemistry
            self.m_n_star = 2 * 1e-5
            self.m_p_star = 6 * 1e-7

            self.F_star = 96487
            self.t_plus = 0.4
            self.Rg_star = 8.314

            # Density
            self.rho_cn_star = 8954
            self.rho_n_star = 1657
            self.rho_s_star = 397
            self.rho_p_star = 3262
            self.rho_cp_star = 2707

            # Specific heat
            self.cp_cn_star = 385
            self.cp_n_star = 700
            self.cp_s_star = 700
            self.cp_p_star = 700
            self.cp_cp_star = 897

            # Thermal conductivity
            self.lambda_cn_star = 401
            self.lambda_n_star = 1.7
            self.lambda_s_star = 0.16
            self.lambda_p_star = 2.1
            self.lambda_cp_star = 237

            # Thermal
            self.T_inf_star = 298.15
            self.h_star = 10
            self.h_tab_star = self.h_star

            # Typical voltage drop
            self.Phi_star = 1

            # Typical temperature difference
            self.Delta_T_star = self.I_star * self.Phi_star / self.h_star

            # Initial conditions
            self.c_e_typ_star = 1e3
            self.c_n_0_star = 19986
            self.c_p_0_star = 40974
            self.T_0_star = self.T_inf_star
        else:
            raise ValueError("Paramters set name not recognised!")

        # Effective material properties
        self.rho_eff_star = (
            self.rho_cn_star * self.cp_cn_star * self.L_cn_star
            + self.rho_n_star * self.cp_n_star * self.L_n_star
            + self.rho_s_star * self.cp_s_star * self.L_s_star
            + self.rho_p_star * self.cp_p_star * self.L_p_star
            + self.rho_cp_star * self.cp_cp_star * self.L_cp_star
        ) / self.L_star
        self.lambda_eff_star = (
            self.lambda_cn_star * self.L_cn_star
            + self.lambda_n_star * self.L_n_star
            + self.lambda_s_star * self.L_s_star
            + self.lambda_p_star * self.L_p_star
            + self.lambda_cp_star * self.L_cp_star
        ) / self.L_star

        # Calculate timescales
        self.tau_d_star = self.F_star * self.c_n_max_star * self.L_star / self.I_star
        self.tau_n_star = self.R_n_star ** 2 / self.D_n_tilde_star
        self.tau_p_star = self.R_p_star ** 2 / self.D_p_tilde_star
        self.tau_e_star = self.L_star ** 2 / self.D_e_typ_star
        self.tau_rn_star = (
            self.F_star / self.m_n_star / self.a_n_star / self.c_e_typ_star ** 0.5
        )
        self.tau_rp_star = (
            self.F_star / self.m_p_star / self.a_p_star / self.c_e_typ_star ** 0.5
        )
        self.tau_th_star = self.rho_eff_star * self.Lz_star ** 2 / self.lambda_eff_star

        # Calculate dimensionless parameters
        self.L_cn = self.L_cn_star / self.Lx_star
        self.L_n = self.L_n_star / self.Lx_star
        self.L_s = self.L_s_star / self.Lx_star
        self.L_p = self.L_p_star / self.Lx_star
        self.L_cp = self.L_cp_star / self.Lx_star

        self.L = self.L_cn + self.L_n + self.L_s + self.L_p + self.L_cp

        self.Ly = self.Ly_star / self.Lz_star

        self.tab_offset = self.tab_offset_star / self.Lz_star

        self.L_tab_n = self.L_tab_n_star / self.Lz_star
        self.A_tab_n = self.A_tab_n_star / self.Lx_star / self.Lz_star
        self.tab_n_centre = self.tab_offset + self.L_tab_n / 2

        self.L_tab_p = self.L_tab_p_star / self.Lz_star
        self.A_tab_p = self.A_tab_p_star / self.Lx_star / self.Lz_star
        self.tab_p_centre = self.Ly - self.tab_offset - self.L_tab_p / 2

        self.gamma_n = self.tau_d_star / self.tau_n_star
        self.gamma_p = self.tau_d_star / self.tau_p_star

        self.m_n = self.tau_d_star / self.tau_rn_star
        self.m_p = self.tau_d_star / self.tau_rp_star

        self.sigma_cn = self.sigma_cn_star * self.Phi_star / self.I_star / self.Lx_star
        self.sigma_n = self.sigma_n_star * self.Phi_star / self.I_star / self.Lx_star
        self.sigma_p = self.sigma_p_star * self.Phi_star / self.I_star / self.Lx_star
        self.sigma_cp = self.sigma_cp_star * self.Phi_star / self.I_star / self.Lx_star

        self.beta_n = self.a_n_star * self.R_n_star
        self.beta_p = self.a_p_star * self.R_p_star

        self.C_hat_n = self.c_n_max_star / self.c_n_max_star
        self.C_hat_p = self.c_p_max_star / self.c_n_max_star

        self.rho_cn = self.rho_cn_star * self.cp_cn_star / self.rho_eff_star
        self.rho_n = self.rho_n_star * self.cp_n_star / self.rho_eff_star
        self.rho_s = self.rho_s_star * self.cp_s_star / self.rho_eff_star
        self.rho_p = self.rho_p_star * self.cp_p_star / self.rho_eff_star
        self.rho_cp = self.rho_cp_star * self.cp_cp_star / self.rho_eff_star

        self.lambda_cn = self.lambda_cn_star / self.lambda_eff_star
        self.lambda_n = self.lambda_n_star / self.lambda_eff_star
        self.lambda_s = self.lambda_s_star / self.lambda_eff_star
        self.lambda_p = self.lambda_p_star / self.lambda_eff_star
        self.lambda_cp = self.lambda_cp_star / self.lambda_eff_star

        self.epsilon = self.Lx_star / self.Lz_star

        self.delta = self.tau_e_star / self.tau_d_star

        self.Lambda = self.F_star * self.Phi_star / self.Rg_star / self.T_inf_star

        self.nu = self.c_n_max_star / self.c_e_typ_star

        self.Theta = self.Delta_T_star / self.T_inf_star

        self.B = (
            self.I_star
            * self.Phi_star
            * self.tau_th_star
            / self.rho_eff_star
            / self.Delta_T_star
            / self.Lx_star
        )

        self.gamma_th = self.tau_d_star / self.tau_th_star

        self.h = self.h_star * self.Lx_star / self.lambda_eff_star
        self.h_tab = self.h_tab_star * self.Lx_star / self.lambda_eff_star

        self.c_n_0 = self.c_n_0_star / self.c_n_max_star
        self.c_p_0 = self.c_p_0_star / self.c_p_max_star
        self.T_0 = (self.T_0_star - self.T_inf_star) / self.Delta_T_star

        # Scaled parameters
        self.sigma_cn_prime = self.sigma_cn * self.epsilon ** 2
        self.sigma_n_prime = self.sigma_n * self.delta
        self.sigma_p_prime = self.sigma_p * self.delta
        self.sigma_cp_prime = self.sigma_cp * self.epsilon ** 2

        self.sigma_cn_dblprime = self.sigma_cn_prime * self.delta
        self.sigma_cp_dblprime = self.sigma_cp_prime * self.delta

        self.alpha = 1 / (self.sigma_cn_prime * self.L_cn) + 1 / (
            self.sigma_cp_prime * self.L_cp
        )
        self.alpha_prime = 1 / (self.sigma_cn_dblprime * self.L_cn) + 1 / (
            self.sigma_cp_dblprime * self.L_cp
        )

        self.h_prime = self.h / (self.epsilon ** 2)
        self.h_tab_prime = self.h_tab / (self.epsilon ** 2)

        # x-averaged values for density and thermal conductivity
        # note lambda is renamed to lambda_x in code
        self.rho = (
            self.rho_cn * self.L_cn
            + self.rho_n * self.L_n
            + self.rho_s * self.L_s
            + self.rho_p * self.L_p
            + self.rho_cp * self.L_cp
        ) / self.L

        self.lambda_x = (
            self.lambda_cn * self.L_cn
            + self.lambda_n * self.L_n
            + self.lambda_s * self.L_s
            + self.lambda_p * self.L_p
            + self.lambda_cp * self.L_cp
        ) / self.L

    def solid_diffusivity_n(self, c):
        """
        Calculates the solid diffusivity in the negative electrode particle as
        a function of concentration.

        Parameters
        ----------
        c: array_like
            Array of concentration in each volume.

        Returns
        -------
        array_like
            The the value of the diffusivity at each given concentration.
        """
        return 1

    def solid_diffusivity_p(self, c):
        """
        Calculates the solid diffusivity in the positive electrode particle as
        a function of concentration.

        Parameters
        ----------
        c: array_like
            Array of concentration in each volume.

        Returns
        -------
        array_like
            The the value of the diffusivity at each given concentration.
        """
        return 1

    def electrolyte_diffusivity(self, c):
        """
        Calculates the electrolyte diffusivity as a function of concentration.

        Parameters
        ----------
        c: array_like
            Array of concentration in each volume.

        Returns
        -------
        array_like
            The the value of the diffusivity at each given concentration.
        """
        if self.name == "mypouch":
            # From LIONSIMBA at ambient temperature
            # Make c dimensional
            c = c * self.c_e_typ_star
            exponent = (
                -4.43 - (54 / (self.T_inf_star - 229 - 5 * 1e-3 * c)) - 0.22 * 1e-3 * c
            )
            D_e = 1e-4 * 10 ** exponent

            # Make D_e dimensionless
            D_e = D_e / self.D_e_typ_star  # Make dimensionless
        else:
            raise ValueError("Paramters set name not recognised!")
        return D_e

    def electrolyte_conductivity(self, c):
        """
        Calculates the electrolyte conductivity as a function of concentration.

        Parameters
        ----------
        c: array_like
            Array of concentration in each volume.

        Returns
        -------
        array_like
            The the value of the conductivity at each given concentration.
        """
        if self.name == "mypouch":
            # From LIONSIMBA at ambient temperature
            # Make c dimensional
            c = c * self.c_e_typ_star

            temp = (
                -10.5
                + 0.668 * 1e-3 * c
                + 0.494 * 1e-6 * c ** 2
                + (0.074 - 1.78 * 1e-5 * c - 8.86 * 1e-10 * c ** 2) * self.T_inf_star
                + (-6.96 * 1e-5 + 2.8 * 1e-8 * c) * self.T_inf_star ** 2
            )
            kappa_e = 1e-4 * c * temp ** 2

            # Make kappa_e dimensionless
            kappa_e = (
                kappa_e
                * self.Rg_star
                * self.T_inf_star
                / self.F_star ** 2
                / self.D_e_typ_star
                / self.c_e_typ_star
            )
        else:
            raise ValueError("Paramters set name not recognised!")
        return kappa_e
