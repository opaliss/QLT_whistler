import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import os
import scipy
from QLT_equations.obliqueQLT import dispersion_relation, dKperpdt, dTperpdt, dKpardt, dTpardt, dBdt, dEdt, dVdt


def get_omega_vec(k_perp, k_par, omega_pe, omega_pi, v_0, alpha_i, alpha_c_perp, alpha_c_par, n_c, omega_0,
                  m_star=-1, ic1=0.5 * 0.99 + 1e-3j, ic2=0.5 * 0.99 + 1e-5j, tol=1e-15):
    """

    :param ic2:
    :param ic1:
    :param k_perp:
    :param k_par:
    :param omega_pe:
    :param omega_pi:
    :param v_0:
    :param alpha_i:
    :param alpha_c_perp:
    :param alpha_c_par:
    :param n_c:
    :param omega_0:
    :param tol:
    :return:
    """
    omega_vec = np.zeros(len(k_perp), dtype="complex128")
    for ii, kk in enumerate(k_perp):
        try:
            omega_vec[ii] = scipy.optimize.newton(dispersion_relation(k_perp=k_perp[ii], k_par=k_par[ii],
                                                                      omega_pe=omega_pe, omega_pi=omega_pi,
                                                                      omega_0=omega_0, v_0=v_0,
                                                                      alpha_c_perp=alpha_c_perp, alpha_i=alpha_i,
                                                                      alpha_c_par=alpha_c_par, n_c=n_c,
                                                                      m_star=m_star), x0=ic1, tol=tol)
        except:
            try:
                omega_vec[ii] = scipy.optimize.newton(dispersion_relation(k_perp=k_perp[ii], k_par=k_par[ii],
                                                                          omega_pe=omega_pe, omega_pi=omega_pi,
                                                                          omega_0=omega_0, v_0=v_0,
                                                                          alpha_c_perp=alpha_c_perp, alpha_i=alpha_i,
                                                                          alpha_c_par=alpha_c_par, n_c=n_c,
                                                                          m_star=m_star), x0=ic2, tol=tol)
            except:
                print("|k| = ", str(np.sqrt(k_par[ii]**2 + k_perp[ii]**2)))

        if np.abs(omega_vec[ii].imag) > 0.01:
            omega_vec[ii] = 0.5
        if np.abs(omega_vec[ii].imag) < -0.01:
            omega_vec[ii] = 0.5
    return omega_vec


def dydt(t, f, k_perp, k_par, omega_pe, omega_pi, k_0, alpha_i, n_c, dk_perp, dk_par,
         omega_0, m_star, ic1, ic2, folder_name="oblique_gamma", plot_=False):
    """

    :param t:
    :param f:
    :param k_perp:
    :param k_par:
    :param omega_pe:
    :param omega_pi:
    :param k_0:
    :param alpha_i:
    :param n_c:
    :param dk_perp:
    :param dk_par:
    :param omega_0:
    :param m_star:
    :param ic1:
    :param ic2:
    :param folder_name:
    :return:
    """
    if len(np.where(f[6:] < 0)) > 1:
        # check where dE becomes negative
        print("negative dE was found")
        f[6:][np.where(f[6:] < 0)] = np.zeros(len(np.where(f[6:] < 0)))


    # dispersion solver
    omega_vec = get_omega_vec(k_perp=k_perp, k_par=k_par, omega_pe=omega_pe, omega_pi=omega_pi, v_0=np.sqrt(f[5]),
                              alpha_i=alpha_i, alpha_c_perp=np.sqrt(2 * f[2]), alpha_c_par=np.sqrt(2 * f[3]), n_c=n_c,
                              omega_0=omega_0, m_star=m_star, ic1=ic1, ic2=ic2)

    if plot_:
        if os.path.exists("/Users/oissan/PycharmProjects/QLT_whistler/figs/secondary_QLT/"
                          + str(folder_name) + "/t_" + str(round(t))+ ".png") is False:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.scatter(np.sqrt(k_perp**2 + k_par**2), omega_vec.imag, linewidth=2)
            ax.set_ylabel(r"$\gamma/|\Omega_{ce}|$", rotation=90)
            ax.set_xlabel(r"$|\vec{k}|d_{e}$")
            ax.set_title("$t = $" + str(round(t)))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.grid(alpha=0.5)
            plt.tight_layout()
            plt.savefig("/Users/oissan/PycharmProjects/QLT_whistler/figs/secondary_QLT/" + str(folder_name) + "/t_" + str(round(t)) + "_imag.png", dpi=300, bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.scatter(np.sqrt(k_perp**2 + k_par**2), omega_vec.real, linewidth=2)
            ax.set_ylabel(r"$\gamma/|\Omega_{ce}|$", rotation=90)
            ax.set_xlabel(r"$|\vec{k}|d_{e}$")
            ax.set_title("$t = $" + str(round(t)))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.grid(alpha=0.5)
            plt.tight_layout()
            plt.savefig("/Users/oissan/PycharmProjects/QLT_whistler/figs/secondary_QLT/" + str(folder_name) + "/t_" + str(round(t)) + "_real.png", dpi=300, bbox_inches='tight')
            plt.close()

    # cold electron kinetic energy
    rhs_K_perp = dKperpdt(E_vec=f[6:], omega_pe=omega_pe, alpha_c_par=np.sqrt(2 * f[3]),
                          alpha_c_perp=np.sqrt(2 * f[2]), n_c=n_c, k_par=k_par, k_perp=k_perp, omega_vec=omega_vec,
                          dk_perp=dk_perp, dk_par=dk_par)

    rhs_K_par = dKpardt(E_vec=f[6:], omega_pe=omega_pe, alpha_c_par=np.sqrt(2 * f[3]), alpha_c_perp=np.sqrt(2 * f[2]),
                        n_c=n_c, k_par=k_par, k_perp=k_perp, omega_vec=omega_vec, dk_perp=dk_perp, dk_par=dk_par)


    # rhs_K_perp_ = dKperpdt(E_vec=f[6:], omega_pe=omega_pe, alpha_c_par=np.sqrt(2 * f[3]),
    #                       alpha_c_perp=np.sqrt(2 * f[2]), n_c=n_c, k_par=k_par, k_perp=k_perp, omega_vec=omega_vec.real,
    #                       dk_perp=dk_perp, dk_par=dk_par)
    #
    # rhs_K_par_ = dKpardt(E_vec=f[6:], omega_pe=omega_pe, alpha_c_par=np.sqrt(2 * f[3]), alpha_c_perp=np.sqrt(2 * f[2]),
    #                     n_c=n_c, k_par=k_par, k_perp=k_perp, omega_vec=omega_vec.real, dk_perp=dk_perp, dk_par=dk_par)

    # cold electron temperature
    rhs_T_perp = dTperpdt(E_vec=f[6:], omega_pe=omega_pe, alpha_c_par=np.sqrt(2 * f[3]), alpha_c_perp=np.sqrt(2 * f[2]),
                          k_par=k_par, k_perp=k_perp, omega_vec=omega_vec, dk_perp=dk_perp, dk_par=dk_par)

    rhs_T_par = dTpardt(E_vec=f[6:], omega_pe=omega_pe, alpha_c_par=np.sqrt(2 * f[3]), alpha_c_perp=np.sqrt(2 * f[2]),
                        k_par=k_par, k_perp=k_perp, omega_vec=omega_vec, dk_perp=dk_perp, dk_par=dk_par)

    # magnetic energy whistler
    rhs_B = dBdt(omega_0=omega_0, k_0=k_0,
                 E_vec=f[6:], omega_pe=omega_pe,
                 alpha_c_par=np.sqrt(2 * f[3]),
                 alpha_c_perp=np.sqrt(2 * f[2]),
                 n_c=n_c, k_par=k_par, k_perp=k_perp,
                 omega_vec=omega_vec, dk_perp=dk_perp, dk_par=dk_par)

    # drift magnitude of cold electrons
    rhs_V = dVdt(omega_0=omega_0, k_0=k_0, E_vec=f[6:], omega_pe=omega_pe, alpha_c_par=np.sqrt(2 * f[3]),
                 alpha_c_perp=np.sqrt(2 * f[2]), n_c=n_c, k_par=k_par, k_perp=k_perp, omega_vec=omega_vec,
                 dk_perp=dk_perp, dk_par=dk_par)

    # electrostatic electric energy
    rhs_E = dEdt(gamma=omega_vec.imag, E_vec=f[6:])

    print("t = ", t)
    print("max gamma = ", np.max(omega_vec.imag))
    return np.concatenate(([rhs_K_perp], [rhs_K_par], [rhs_T_perp], [rhs_T_par], [rhs_B], [rhs_V], rhs_E))


if __name__ == "__main__":
    # parameters from 2021 paper
    # normalization (vadim parameters)
    # time is normalized to the electron cyclotron frequency 1/Omega_ce
    # space is normalized to electron inertial length d_e
    omega_0 = 0.5  # Omega_ce
    omega_pe = 4  # Omega_{ce}

    n_c = 4 / 5  # n^e_0

    # thermal velocity
    alpha_c_par = 0.0079  # d_e x Omega_ce
    alpha_c_perp = 0.0079  # d_e x Omega_ce
    alpha_i = alpha_c_par / np.sqrt(1836)  # d_e x Omega_ce

    v_0 = 0.65 * alpha_c_par  # d_e x Omega_ce
    omega_pi = omega_pe / np.sqrt(1836)  # Omega_ce

    # initial conditions
    E0 = 1e-9
    K_perp_0 = ((alpha_c_perp ** 2) / 2) * n_c
    K_par_0 = ((alpha_c_par ** 2) / 2) * n_c
    T_perp_0 = ((alpha_c_perp ** 2) / 2)
    T_par_0 = ((alpha_c_par ** 2) / 2)
    k_0 = 1  # d_e
    dB0 = 4 * np.pi * 2.5 * 1e-5  # d_{e}^3 Omega_{ce}^2 m_{e} n_{e}

    m_star = -1
    ic1 = 0.5 * 0.99 + 1e-3j
    ic2 = 0.5 * 0.99 + 1e-5j

    # max time
    t_max = 600

    k_perp_ = np.linspace(10, 50, 50)
    k_par_ = np.sqrt((omega_0 ** 2) / (1 - omega_0 ** 2)) * k_perp_
    sol_ = np.zeros((len(k_perp_)), dtype="complex128")
    k_abs = np.zeros((len(k_perp_)))
    dk_perp = np.abs(k_perp_[1] - k_perp_[0])
    dk_par = np.abs(k_par_[1] - k_par_[0])
    dE_init = E0 * np.ones(len(k_perp_))

    # simulate
    result = scipy.integrate.solve_ivp(fun=dydt, t_span=[0, t_max],
                                       y0=np.concatenate(([K_perp_0], [K_par_0], [T_perp_0], [T_par_0], [dB0], [v_0 ** 2], dE_init)),
                                       args=(k_perp_, k_par_, omega_pe, omega_pi, k_0, alpha_i, n_c, dk_perp, dk_par,
                                             omega_0, m_star, ic1, ic2, "oblique_gamma", False),
                                       atol=1e-9, rtol=1e-9,
                                       method='Radau')

    np.save("oblique_y.npy", result.y)
    np.save("oblique_t.npy", result.t)
