% Script to calcuate non-dimensional parameters as a function of C-rate
% Parameter values taken from Scott Moura's DFN (and SPMe with temperature)
% available on github: 
% https://github.com/scott-moura/SPMeT/blob/master/param/params_LCO.m
close all
clear all
clc

%% C-rate
C_rate = 1;

%% Dimensional parameters
L_cn_star = 25*1e-6;
L_n_star = 100*1e-6;
L_s_star = 25*1e-6;
L_p_star = 100*1e-6;
L_cp_star = 25*1e-6;

L_tab_n_star = 40*1e-3;
L_tab_p_star = 40*1e-3;
tab_offset_star = 30*1e-3;

A_n_star = L_cn_star*L_tab_n_star;
A_p_star = L_cp_star*L_tab_n_star;

epsilon_s_n = 0.6;
epsilon_s_p = 0.5; 

epsilon_n = 0.3;
epsilon_s = 1;
epsilon_p = 0.3;

epsilon_f_n = 1 - epsilon_s_n - epsilon_n;
epsilon_f_p = 1 - epsilon_s_p - epsilon_p;


c_n_max_star = 2.4983*1e4;
c_p_max_star = 5.1218*1e4;

sigma_cn_star = 5.96*1e7;
sigma_n_star = 100;
sigma_p_star = 10;
sigma_cp_star = 3.55*1e7;

D_n_tilde_star = 3.9*1e-14;
D_p_tilde_star = 1*1e-13;

R_n_star = 10*1e-6;
R_p_star = 10*1e-6;

a_n_star = 3 * epsilon_s_n / R_n_star;
a_p_star = 3 * epsilon_s_p / R_p_star;

m_n_star = 2*1e-5;
m_p_star = 6*1e-7;

rho_sn_star = 1800;
rho_sp_star = 5010;
rho_e_star = 1324;
rho_f_star = 1800;

rho_cn_star = 8954;
rho_n_star = rho_e_star * epsilon_n ...
             + rho_sn_star * epsilon_s_n ...
             + rho_f_star * epsilon_f_n;
rho_s_star = rho_e_star * epsilon_n;
rho_p_star = rho_e_star * epsilon_p ...
             + rho_sp_star * epsilon_s_p ...
             + rho_f_star * epsilon_f_p;
rho_cp_star = 2707;

cp_cn_star = 385;
cp_n_star = 700;
cp_s_star = 700;
cp_p_star = 700;
cp_cp_star = 897;

lambda_cn_star = 401; 
lambda_n_star = 1.7;
lambda_s_star = 0.16;
lambda_p_star = 2.1;
lambda_cp_star = 237;


c_e_typ_star = 1e3;

D_e_typ_star = 5.34*1e-10;

F_star = 96487;

Rg_star = 8.314;

T_inf_star = 298.15;

b = 1.5;

t_plus = 0.4;

Lx_star =  L_n_star +  L_s_star + L_p_star;
Ly_star = 180*1e-3;
Lz_star = 220*1e-3;

Phi_star = 1; %Rg_star*T_inf_star/F_star;

%I_app_1C = 2.3;
I_star = C_rate * 24; %C_rate * I_app_1C / (Ly_star*Lz_star);
I_app_1C = I_star / C_rate * (Ly_star*Lz_star);

h_star = 10;
h_tab_star = 10;


c_n_0_star = 0.8*c_n_max_star;
c_p_0_star = 0.6*c_p_max_star;

T_0_star = T_inf_star;

%
% Delta T computed from balance with surface heat loss
Delta_T_star = I_star*Phi_star/h_star;
%


L_star = L_cn_star + L_n_star +  L_s_star + L_p_star + L_cp_star;

%% Effective material properties

rho_eff_star = (rho_cn_star*cp_cn_star*L_cn_star ...
    + rho_n_star*cp_n_star*L_n_star ...
    + rho_s_star*cp_s_star*L_s_star ...
    + rho_p_star*cp_p_star*L_p_star ...
    + rho_cp_star*cp_cp_star*L_cp_star)/L_star;

lambda_eff_star = (lambda_cn_star*L_cn_star ...
    + lambda_n_star*L_n_star ...
    + lambda_s_star*L_s_star ...
    + lambda_p_star*L_p_star ...
    + lambda_cp_star*L_cp_star)/L_star;

%% Timescales

tau_d_star = F_star*c_n_max_star*L_star/I_star;

tau_n_star = (R_n_star)^2/D_n_tilde_star;
tau_p_star = (R_p_star)^2/D_p_tilde_star;

tau_e_star = (L_star)^2/D_e_typ_star;

tau_rn_star = F_star/(m_n_star*a_n_star*(c_e_typ_star)^0.5);
tau_rp_star = F_star/(m_p_star*a_p_star*(c_e_typ_star)^0.5);

tau_th_star = (rho_eff_star*Lz_star^2)/lambda_eff_star;

%% Non-dimensional parameters

L_cn = L_cn_star/Lx_star;
L_n = L_n_star/Lx_star;
L_s = L_s_star/Lx_star;
L_p = L_p_star/Lx_star;
L_cp = L_cp_star/Lx_star;

A_n = A_n_star/(Lx_star*Lz_star);
A_p = A_p_star/(Lx_star*Lz_star);

gamma_n = tau_d_star/tau_n_star;
gamma_p = tau_d_star/tau_p_star;

m_n = tau_d_star/tau_rn_star;
m_p = tau_d_star/tau_rp_star;

sigma_cn = (sigma_cn_star*Phi_star)/(I_star*Lx_star);
sigma_n = (sigma_n_star*Phi_star)/(I_star*Lx_star);
sigma_p = (sigma_p_star*Phi_star)/(I_star*Lx_star);
sigma_cp = (sigma_cp_star*Phi_star)/(I_star*Lx_star);

beta_n = a_n_star*R_n_star;
beta_p = a_p_star*R_p_star;

C_hat_n = c_n_max_star/c_n_max_star;
C_hat_p = c_p_max_star/c_n_max_star;

rho_cn = rho_cn_star*cp_cn_star/(rho_eff_star);
rho_n = rho_n_star*cp_n_star/(rho_eff_star);
rho_s = rho_s_star*cp_s_star/(rho_eff_star);
rho_p = rho_p_star*cp_p_star/(rho_eff_star);
rho_cp = rho_cp_star*cp_cp_star/(rho_eff_star);

lambda_cn = lambda_cn_star/lambda_eff_star;
lambda_n = lambda_n_star/lambda_eff_star;
lambda_s = lambda_s_star/lambda_eff_star;
lambda_p = lambda_p_star/lambda_eff_star;
lambda_cp = lambda_cp_star/lambda_eff_star;


epsilon = Lx_star/Lz_star;

L_y = Ly_star/Lz_star;

delta = tau_e_star/tau_d_star;

Lambda = F_star*Phi_star/(Rg_star*T_inf_star);

nu = c_n_max_star/c_e_typ_star;

Theta = Delta_T_star/T_inf_star;

B = I_star*Phi_star*tau_th_star/(rho_eff_star*Delta_T_star*Lx_star);

gamma_th = tau_d_star/tau_th_star;

h = h_star*Lx_star/lambda_eff_star;
h_tab = h_tab_star*Lx_star/lambda_eff_star;


c_n_0 = c_n_0_star/c_n_max_star;

c_p_0 = c_p_0_star/c_p_max_star;

T_0 = (T_0_star - T_inf_star)/Delta_T_star;



% alpha
alpha = 1/(epsilon^2*sigma_cn*L_cn) + 1/(epsilon^2*sigma_cp*L_cp);
alpha_prime = 1/(epsilon^2*delta*sigma_cn*L_cn) + 1/(epsilon^2*delta*sigma_cp*L_cp);

% x-averaged values for density and thermal conductivity 
rho = (rho_cn*L_cn ...
    + rho_n*L_n ...
    + rho_s*L_s ...
    + rho_p*L_p ...
    + rho_cp*L_cp)/(1 + L_cn + L_cp);

lambda = (lambda_cn*L_cn ...
    + lambda_n*L_n ...
    + lambda_s*L_s ...
    + lambda_p*L_p ...
    + lambda_cp*L_cp)/(1 + L_cn + L_cp);
















