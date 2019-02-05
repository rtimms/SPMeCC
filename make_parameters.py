class Parameters(object):

    def __init__(self, C_rate=1, name='mypouch'):
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
        self.C_rate = 1

        # Put name in parameters object
        self.name = name

        # Load up dimensional parameters
        if self.name == 'mypouch':
            # Geometry
            self.L_cn_star = 15*1E-6
            self.L_n_star = 63.4*1E-6
            self.L_s_star = 25*1E-6
            self.L_p_star = 55.3*1E-6
            self.L_cp_star = 20*1E-6

            self.L_tab_n_star = 40*1E-3
            self.L_tab_p_star = 40*1E-3
            self.tab_offset_star = 30*1E-3
            self.tab_n_location = 't'  # Location of negative tab (t, b, l, r)
            self.tab_p_location = 't'  # Location of negative tab (t, b, l, r)

            self.A_tab_n_star = self.L_cn_star * self.L_tab_n_star
            self.A_tab_p_star = self.L_cp_star * self.L_tab_n_star

            self.Lx_star = self.L_n_star + self.L_s_star + self.L_p_star
            self.Ly_star = 180*1E-3
            self.Lz_star = 220*1E-3

            self.L_star = self.L_cn_star + self.Lx_star + self.L_cp_star

            # Porosity
            self.epsilon_n = 0.485
            self.epsilon_s = 0.724
            self.epsilon_p = 0.385
            self.brug = 4

            # Filler fractions
            self.epsilon_f_n = 0.0326
            self.epsilon_f_p = 0.025

            # Typical voltage drop
            self.Phi_star = 1

            # Cutoff voltage
            self.V_min = 2.5

            # Cutover voltage
            self.V_max = 4.3

            # Applied current density
            # self.I_app_1C = 2.3
            # self.I_star = (self.C_rate * self.I_app_1C
            #                / (self.Ly_star * self.Lz_star))
            # Fudge to make leading-order SPMeCC current density equivalent
            # to that applied in LIONSIMBA
            self.I_star = 29.5 * (self.Ly_star / self.Lz_star)

            # Electrical conductivity
            self.sigma_cn_star = 5.96*1E7
            self.sigma_n_star = 100
            self.sigma_p_star = 100
            self.sigma_cp_star = 3.55*1E7

            # Rescale conductivity as in LIONSIMBA
            self.sigma_n_star = (self.sigma_n_star
                                 * (1 - self.epsilon_n - self.epsilon_f_n))
            self.sigma_p_star = (self.sigma_p_star
                                 * (1 - self.epsilon_p - self.epsilon_f_p))

            # Diffusivity
            self.D_n_tilde_star = 3.9*1E-14
            self.D_p_tilde_star = 1*1E-14
            self.D_e_typ_star = 7.5*1E-10

            # Particle
            self.c_n_max_star = 30555
            self.c_p_max_star = 51554
            self.R_n_star = 2*1E-6
            self.R_p_star = 2*1E-6
            self.a_n_star = 723600
            self.a_p_star = 885000

            # Electrochemistry
            self.m_n_star = 2 * 5.031*1E-11 * 96487
            self.m_p_star = 2 * 2.334*1E-11 * 96487

            self.F_star = 96487
            self.t_plus = 0.364
            self.Rg_star = 8.314

            # Density
            self.rho_cn_star = 8940
            self.rho_n_star = 2500
            self.rho_s_star = 1100
            self.rho_p_star = 2500
            self.rho_cp_star = 2700

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
            self.h_star = 1
            self.h_tab_star = 1
            self.Delta_T_star = self.I_star * self.Phi_star / self.h_star

            # Initial conditions
            self.c_e_typ_star = 1e3
            self.c_n_0_star = 26128
            self.c_p_0_star = 25751
            self.T_0_star = self.T_inf_star

        else:
            raise ValueError('Paramters set name not recognised!')

        # Effective material properties
        self.rho_eff_star = (self.rho_cn_star * self.cp_cn_star
                             * self.L_cn_star
                             + self.rho_n_star * self.cp_n_star
                             * self.L_n_star
                             + self.rho_s_star * self.cp_s_star
                             * self.L_s_star
                             + self.rho_p_star * self.cp_p_star
                             * self.L_p_star
                             + self.rho_cp_star * self.cp_cp_star
                             * self.L_cp_star) / self.L_star
        self.lambda_eff_star = ((self.lambda_cn_star * self.L_cn_star
                                 + self.lambda_n_star * self.L_n_star
                                 + self.lambda_s_star * self.L_s_star
                                 + self.lambda_p_star * self.L_p_star
                                 + self.lambda_cp_star * self.L_cp_star)
                                / self.L_star)

        # Calculate timescales
        self.tau_d_star = (self.F_star * self.c_n_max_star * self.L_star
                           / self.I_star)
        self.tau_n_star = self.R_n_star ** 2 / self.D_n_tilde_star
        self.tau_p_star = self.R_p_star ** 2 / self.D_p_tilde_star
        self.tau_e_star = self.L_star ** 2 / self.D_e_typ_star
        self.tau_rn_star = (self.F_star / self.m_n_star / self.a_n_star
                            / self.c_e_typ_star ** 0.5)
        self.tau_rp_star = (self.F_star / self.m_p_star / self.a_p_star
                            / self.c_e_typ_star ** 0.5)
        self.tau_th_star = (self.rho_eff_star * self.Lz_star ** 2
                            / self.lambda_eff_star)

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

        self.sigma_cn = (self.sigma_cn_star * self.Phi_star
                         / self.I_star / self.Lx_star)
        self.sigma_n = (self.sigma_n_star * self.Phi_star
                        / self.I_star / self.Lx_star)
        self.sigma_p = (self.sigma_p_star * self.Phi_star
                        / self.I_star / self.Lx_star)
        self.sigma_cp = (self.sigma_cp_star * self.Phi_star
                         / self.I_star / self.Lx_star)

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

        self.Lambda = (self.F_star * self.Phi_star
                       / self.Rg_star / self.T_inf_star)

        self.nu = self.c_n_max_star / self.c_e_typ_star

        self.Theta = self.Delta_T_star / self.T_inf_star

        self.B = (self.I_star * self.Phi_star * self.tau_th_star
                  / self.rho_eff_star / self.Delta_T_star / self.Lx_star)

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

        self.alpha = (1 / (self.sigma_cn_prime * self. L_cn)
                      + 1 / (self.sigma_cp_prime * self.L_cp))
        self.alpha_prime = (1 / (self.sigma_cn_dblprime * self.L_cn)
                            + 1 / (self.sigma_cp_dblprime * self.L_cp))

        self.h_prime = self.h / (self.epsilon ** 2)
        self.h_tab_prime = self.h_tab / (self.epsilon ** 2)

        # x-averaged values for density and thermal conductivity
        # note lambda is renamed to lambda_x in code
        self.rho = (self.rho_cn * self.L_cn
                    + self.rho_n * self.L_n
                    + self.rho_s * self.L_s
                    + self.rho_p * self.L_p
                    + self.rho_cp * self.L_cp) / self.L

        self.lambda_x = (self.lambda_cn * self.L_cn
                         + self.lambda_n * self.L_n
                         + self.lambda_s * self.L_s
                         + self.lambda_p * self.L_p
                         + self.lambda_cp * self.L_cp) / self.L

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
        if self.name == 'mypouch':
            # From LIONSIMBA at ambient temperature
            # Make c dimensional
            c = c * self.c_e_typ_star
            exponent = (-4.43 - (54 / (self.T_inf_star - 229 - 5 * 1E-3 * c))
                        - 0.22 * 1E-3 * c)
            D_e = 1E-4 * 10 ** exponent

            # Make D_e dimensionless
            D_e = D_e / self.D_e_typ_star  # Make dimensionless
        else:
            raise ValueError('Paramters set name not recognised!')
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
        if self.name == 'mypouch':
            # From LIONSIMBA at ambient temperature
            # Make c dimensional
            c = c * self.c_e_typ_star

            temp = (-10.5 + 0.668 * 1E-3 * c + 0.494 * 1E-6 * c ** 2
                    + (0.074 - 1.78 * 1E-5 * c - 8.86 * 1E-10 * c ** 2)
                    * self.T_inf_star
                    + (-6.96 * 1E-5 + 2.8 * 1E-8 * c) * self.T_inf_star ** 2)
            kappa_e = 1E-4 * c * temp ** 2

            # Make kappa_e dimensionless
            kappa_e = (kappa_e * self.Rg_star * self.T_inf_star
                       / self.F_star ** 2 / self.D_e_typ_star
                       / self.c_e_typ_star)
        else:
            raise ValueError('Paramters set name not recognised!')
        return kappa_e


# Paramters from draft. Note that I had made a mistake and the values for
# cp and cn are the wrong way around...
class Parameters_Scott(object):

    def __init__(self, C_rate=1):
        """
        The parameters for the simulation.
        Parameters
        ----------
        C_rate : float
                 The C-rate for the discharge
        """
        # Set default C-rate to be 1
        self.C_rate = C_rate

        # Set other non-dimensional parameters

        self.L_cn = 0.0444  # Negative current collector thickness
        self.L_n = 0.4444  # Negative electrode thickness
        self.L_s = 0.1111  # Separator thickness
        self.L_p = 0.4444  # Positive electrode thickness
        self.L_cp = 0.0444  # Positive current collector thickness

        self.L = self.L_cn + self.L_n + self.L_s + self.L_p + self.L_cp

        self.Ly = 0.8182  # Cell width

        self.L_tab_n = 0.3  # Negative tab width
        self.A_tab_n = self.L_cn * self.L_tab_n  # Negative tab area
        self.tab_n_centre = 0.1 + self.L_tab_n / 2  # Centre point of neg. tab
        self.tab_n_location = 't'  # Location of negative tab (t, b, l, r)

        self.L_tab_p = 0.3  # Positive tab width
        self.A_tab_p = self.L_cp * self.L_tab_p  # Positive tab area
        self.tab_p_centre = (self.Ly - 0.1
                             - self.L_tab_p / 2)  # Centre point of pos. tab
        self.tab_p_location = 't'  # Location of negative tab (t, b, l, r)

        self.gamma_n = 9.5969/self.C_rate
        self.gamma_p = 24.6075/self.C_rate

        self.m_n = 26.6639  # Negative rate constant
        self.m_p = 1.3666  # Positive rate constant

        self.sigma_cn = 6.5741*1E9 / self.C_rate  # Neg. cc conductivity
        self.sigma_n = 1.8519*1E4 / self.C_rate  # Neg. electrode conductivity
        self.sigma_p = 1.8519*1E3 / self.C_rate  # Pos. electrode conductivity
        self.sigma_cp = 1.1037*1E10 / self.C_rate  # Pos. cc conductivity

        self.beta_n = 1.8  # Negative particle radius
        self.beta_p = 1.5  # Positive particle radius

        self.C_hat_n = 1.0  # Ratio c_n_max/c_n_max
        self.C_hat_p = 2.0501  # Ratio c_p_max/c_n_max

        self.rho_cn = 1.3866  # Neg. cc density
        self.rho_n = 1.0019  # Neg. electrode density
        self.rho_s = 0.4408  # Separator density
        self.rho_p = 1.0019  # Pos. electrode density
        self.rho_cp = 1.9728  # Pos. cc density

        self.rho = 1.0

        self.lambda_cn = 8.5844  # Neg. cc thermal conductivity
        self.lambda_n = 0.0761  # Neg. electrode thermal conductivity
        self.lambda_s = 0.0058  # Separator thermal conductivity
        self.lambda_p = 0.0616  # Pos. electrode thermal conductivity
        self.lambda_cp = 14.5247  # Pos. cc thermal conductivity

        self.epsilon = 1E-3  # Aspect ratio

        self.delta = 4.6*1E-3 * self.C_rate

        self.Lambda = 39.9224

        self.nu = 24.9833  # Ratio c_n_max/c_e_typ

        self.B = 77.9518 * self.C_rate

        self.gamma_th = 8.0363 / self.C_rate

        self.Theta = 0.008 * self.C_rate

        self.h = 8.1498*1E-5

        self.h_tab = 8.1498*1E-5

        self.c_n_0 = 0.8
        self.c_p_0 = 0.6

        self.c_e_0 = 1

        self.T_0 = 0

        self.sigma_n_prime = self.sigma_n * self.delta
        self.sigma_p_prime = self.sigma_p * self.delta

        self.sigma_cn_dblprime = self.sigma_cn * self.epsilon ** 2 * self.delta
        self.sigma_cp_dblprime = self.sigma_cp * self.epsilon ** 2 * self.delta

        self.alpha_prime = (1 / (self.sigma_cn_dblprime * self.L_cn)
                            + 1 / (self.sigma_cp_dblprime * self.L_cp))

        self.h_prime = self.h / (self.epsilon ** 2)
        self.h_tab_prime = self.h_tab / (self.epsilon ** 2)

        self.tau_d_star = 2.4608*1E4

        self.epsilon_n = 0.3
        self.epsilon_s = 1.0
        self.epsilon_p = 0.3

        self.brug = 1.5  # Bruggeman porosity

        self.t_plus = 0.4  # transferrance number

        self.Phi_star = 1

        self.Delta_T_star = 2.4
