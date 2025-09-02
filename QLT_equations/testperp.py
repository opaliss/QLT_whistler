import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
import matplotlib

matplotlib.use('TkAgg')
from QLT_equations.perpQLT import dispersion_relation, dVdt, dBdt, dEdt, dKdt


def get_omega_vec(k_vec, omega_pe, omega_pi, v_0, alpha_i, alpha_perp_c, n_c, omega_0,
                  ic1=1.5 + 1E-3j, ic2=1. + 1E-4j, tol=1e-15, maxiter=1000):
    """

    :param tol:
    :param ic2:
    :param ic1:
    :param k_vec:
    :param omega_pe:
    :param omega_pi:
    :param v_0:
    :param alpha_i:
    :param alpha_perp_c:
    :param n_c:
    :param omega_0:
    :return:
    """
    omega_vec = np.zeros(len(k_vec), dtype="complex128")
    for ii in range(len(k_vec)):
        try:
            if ii > 0:
                x0 = ic1
                x1 = ic2
            else:
                x0 = omega_vec[ii - 1]
                x1 = ic1
            omega_vec[ii] = scipy.optimize.newton(dispersion_relation(k_perp=kk, omega_pe=omega_pe, omega_0=omega_0,
                                                                      omega_pi=omega_pi, v_0=v_0, alpha_i=alpha_i,
                                                                      alpha_perp_c=alpha_perp_c, n_c=n_c),
                                                  x0=x0, x1=x1, tol=tol, maxiter=maxiter)
        except:
            try:
                omega_vec[ii] = scipy.optimize.newton(dispersion_relation(k_perp=kk, omega_pe=omega_pe, omega_0=omega_0,
                                                                          omega_pi=omega_pi, v_0=v_0, alpha_i=alpha_i,
                                                                          alpha_perp_c=alpha_perp_c, n_c=n_c),
                                                      x0=ic1, x1=ic2, tol=tol, maxiter=maxiter)
            except:
                omega_vec[ii] = 0
                print("k|_", str(k_vec[ii]))
        if omega_vec[ii].imag > 0.1:
            omega_vec[ii] = omega_vec[ii].real
        if omega_vec[ii].imag < -0.2:
            omega_vec[ii] = omega_vec[ii].real
            print("negative val", k_vec[ii])
    return omega_vec


def dydt(t, f, k_vec, omega_pe, omega_pi, k_0, alpha_i, n_c, dk, omega_0, ic1=1.5 + 1E-3j, ic2=1. + 1E-4j,
         folder_name="perp_gamma", plot=False):
    """

    :param t:
    :param f:
    :param k_vec:
    :param omega_pe:
    :param omega_pi:
    :param k_0:
    :param alpha_i:
    :param n_c:
    :param dk:
    :param omega_0:
    :param folder_name:
    :param plot: boolean, if true then save plots of growth rate
    :return:
    """
    # dispersion solver
    omega_vec = get_omega_vec(k_vec=k_vec, omega_pe=omega_pe, omega_pi=omega_pi,
                              v_0=np.sqrt(f[3]), omega_0=omega_0, alpha_i=alpha_i,
                              alpha_perp_c=np.sqrt(2 * f[1]), n_c=n_c, ic1=ic1, ic2=ic2)

    if plot:
        if os.path.exists("/Users/oissan/PycharmProjects/QLT_whistler/figs/secondary_QLT/"
                          + str(folder_name) + "/t_" + str(round(t)) + ".png") is False:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(k_vec, omega_vec.imag, linewidth=2)
            ax.set_ylabel('$\gamma/|\Omega_{ce}|$', rotation=90)
            ax.set_xlabel(r"$k_{\perp}d_{e}$")
            ax.set_ylim(-0.0005, 0.012)
            ax.set_xlim(176, 220)
            ax.set_title("$t = $" + str(round(t)))
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig("/Users/oissan/PycharmProjects/QLT_whistler/figs/secondary_QLT/" + str(folder_name) + "/t_" + str(
                round(t)) + ".png", dpi=300, bbox_inches='tight')
            plt.close()

    # cold electron kinetic energy
    rhs_K = dKdt(omega_pi=omega_pi, alpha_i=alpha_i, E_vec=f[4:], k_vec=k_vec, omega_vec=omega_vec, dk=dk,
                 omega_0=omega_0, v_0=np.sqrt(f[3]))

    rhs_T = dKdt(omega_pi=omega_pi, alpha_i=alpha_i, E_vec=f[4:], k_vec=k_vec, omega_vec=omega_vec.real, dk=dk,
                 omega_0=omega_0, v_0=np.sqrt(f[3])) / n_c

    # electrostatic electric energy
    rhs_E = dEdt(gamma=omega_vec.imag, E_vec=f[4:])

    # magnetic energy whistler
    rhs_B = dBdt(omega_pi=omega_pi, alpha_i=alpha_i, E_vec=f[4:], k_vec=k_vec, omega_vec=omega_vec, dk=dk,
                 omega_0=omega_0, v_0=np.sqrt(f[3]), k_0=k_0)

    # drift magnitude of cold electrons
    rhs_V = dVdt(omega_0=omega_0, k_0=k_0, omega_pi=omega_pi, alpha_i=alpha_i, E_vec=f[4:], k_vec=k_vec,
                 omega_vec=omega_vec, dk=dk, v_0=np.sqrt(f[3]), omega_pe=omega_pe)

    print("t = ", t)
    print("max gamma = ", np.max(omega_vec.imag))
    return np.concatenate(([rhs_K], [rhs_T], [rhs_B], [rhs_V], rhs_E))


if __name__ == "__main__":
    # normalization
    # time is normalized to the electron cyclotron frequency 1/Omega_ce
    # space is normalized to electron inertial length d_e
    n_c = 4 / 5  # n^e_0
    omega_0 = 0.5  # Omega_ce
    k_0 = 1  # d_e
    dB0 = 4 * np.pi * (5e-5)  # d_{e}^3 Omega_{ce}^2 m_{e} n_{e}

    omega_pe = 4  # Omgea_{ce}
    alpha_perp_c = 0.0079  # d_e x Omega_ce
    alpha_i = 0.0079 / np.sqrt(1836)  # d_e x Omega_ce

    v_0 = 0.65 * 0.0079  # d_e x Omega_ce
    omega_pi = omega_pe / np.sqrt(1836)  # Omega_ce

    # initial conditions
    E0 = 5e-9
    K0 = (alpha_perp_c ** 2 / 2) * n_c  # + n_c * (v_0**2)
    T0 = (alpha_perp_c ** 2 / 2)

    # k vector
    k_vec = np.linspace(176, 220, 150)
    dk = np.abs(k_vec[1] - k_vec[0])

    # max time
    t_max = 600

    dE_init = E0 * np.ones(len(k_vec))

    # simulate
    result = scipy.integrate.solve_ivp(fun=dydt, t_span=[0, t_max],
                                       y0=np.concatenate(([K0], [T0], [dB0], [v_0 ** 2], dE_init)),
                                       args=(k_vec, omega_pe, omega_pi, k_0, alpha_i, n_c, dk, omega_0),
                                       atol=1e-10, rtol=1e-10,
                                       method='BDF')
