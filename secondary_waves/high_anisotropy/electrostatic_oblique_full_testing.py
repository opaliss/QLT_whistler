import numpy as np
from QLT_equations.general_plasma_equations import Z, Z_prime, I, J
import scipy

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def sum_bessel(lambda_, omega, kz, alpha_c_par_, alpha_c_perp_, n_max=20):
    res = 0
    for n in range(-n_max, n_max + 1):
        xi_0 = omega / (kz * alpha_c_par_)
        xi_n = (omega - n) / (kz * alpha_c_par_)
        ani_term = n / kz / alpha_c_par_ * ((alpha_c_par_ ** 2) / (alpha_c_perp_ ** 2) - 1)
        res += I(m=n, Lambda=lambda_) * Z(z=xi_n) * (xi_0 + ani_term)
    return res


def electron_response(n_c_, omega_pe_, alpha_c_perp_, alpha_c_par_, omega, kz, ky, n_max=20):
    # Bessel argument electron
    lambda_ = 0.5 * ((ky * alpha_c_perp_) ** 2)
    return 2 * n_c_ * ((omega_pe_ ** 2) / (alpha_c_par_ ** 2)) * (
            1 + sum_bessel(lambda_=lambda_, omega=omega, kz=kz, n_max=n_max,
                           alpha_c_par_=alpha_c_par_, alpha_c_perp_=alpha_c_perp_))


def THETA(omega_pi_, alpha_i_, n, M, ky, v_0_, omega_0_, omega, kz, m_max=20):
    # magnitude of the wavevector
    k_abs = np.sqrt(ky ** 2 + kz ** 2)
    # Bessel argument ion Doppler-shifted
    a = ky * np.abs(v_0_) / omega_0_
    res = 0
    for m in range(-m_max, m_max + 1):
        res += (omega_pi_ ** 2) / (alpha_i_ ** 2) * J(m=m - n, Lambda=-a) * J(m=m - M, Lambda=-a) \
               * Z_prime(z=(omega + m * omega_0_) / (alpha_i_ * k_abs))
    return res


def D_matrix_opal(omega, ky, kz, n_c_, omega_pe_, alpha_c_par_, alpha_c_perp_, omega_0_, v_0_, alpha_i_, omega_pi_,
                  N=10):
    # initialize matrix
    D_mat = np.zeros((N * 2 + 1, N * 2 + 1), dtype="complex128")
    k2 = ky ** 2 + kz ** 2
    for n in range(-N, N + 1):
        i1 = n + N
        # cold electron components
        D_mat[i1, i1] += k2 + electron_response(n_c_=n_c_, omega_pe_=omega_pe_,
                                                alpha_c_par_=alpha_c_par_,
                                                alpha_c_perp_=alpha_c_perp_, omega=omega + n * omega_0_,
                                                kz=kz, ky=ky)

        for M in range(-N, N + 1):
            i2 = M + N
            D_mat[i1, i2] += -THETA(omega_pi_=omega_pi_, alpha_i_=alpha_i_, n=n, M=M, ky=ky,
                                    v_0_=v_0_, omega_0_=omega_0_, omega=omega, kz=kz)
    return D_mat


def ion_response(omega_pi_, alpha_i_, m_star_, ky, v_0_, omega_0_, omega, kz):
    k_abs = np.sqrt(ky ** 2 + kz ** 2)
    # Bessel argument ion Doppler-shifted
    a = ky * np.abs(v_0_) / omega_0
    return (omega_pi_ ** 2) / (alpha_i_ ** 2) * (J(m=m_star_, Lambda=-a) ** 2) * Z_prime(
        z=(omega + m_star_ * omega_0_) / (alpha_i_ * k_abs))


if __name__ == "__main__":
    # parameters from 2021 paper
    # normalization (Vadim parameters)
    # time is normalized to the electron cyclotron frequency 1/Omega_ce
    # space is normalized to electron inertial length d_e
    omega_0 = 0.5  # Omega_ce
    omega_pe = 4  # Omgea_{ce}

    n_c = 4 / 5  # n^e_0

    # thermal velocity
    alpha_c_par = 0.0079  # d_e x Omega_ce
    alpha_c_perp = 0.0079  # d_e x Omega_ce
    alpha_i = alpha_c_par / np.sqrt(1836)  # d_e x Omega_ce

    v_0 = 0.65 * alpha_c_par  # d_e x Omega_ce
    omega_pi = omega_pe / np.sqrt(1836)  # Omega_ce


    def disp_k_approx(ky,
                      kz,
                      omega_pe_=omega_pe,
                      omega_pi_=omega_pi,
                      omega_0_=omega_0,
                      v_0_=v_0,
                      alpha_i_=alpha_i,
                      alpha_c_perp_=alpha_c_perp,
                      alpha_c_par_=alpha_c_par,
                      n_c_=n_c,
                      m_star=-1):
        return lambda omega: ky ** 2 + kz ** 2 \
                             + electron_response(n_c_=n_c_, omega_pe_=omega_pe_, alpha_c_par_=alpha_c_par_,
                                                 alpha_c_perp_=alpha_c_perp_, omega=omega, kz=kz, ky=ky) \
                             - ion_response(omega_pi_=omega_pi_, alpha_i_=alpha_i_, m_star_=m_star, ky=ky, v_0_=v_0_,
                                            omega_0_=omega_0_, omega=omega, kz=kz)


    def disp_k_full(ky,
                    kz,
                    omega_pe_=omega_pe,
                    omega_pi_=omega_pi,
                    omega_0_=omega_0,
                    v_0_=v_0,
                    alpha_i_=alpha_i,
                    alpha_c_perp_=alpha_c_perp,
                    alpha_c_par_=alpha_c_par,
                    n_c_=n_c):
        return lambda omega: np.linalg.det(
            D_matrix_opal(omega=omega, ky=ky, kz=kz, n_c_=n_c_, omega_pe_=omega_pe_, alpha_c_par_=alpha_c_par_,
                          alpha_c_perp_=alpha_c_perp_, omega_0_=omega_0_, v_0_=v_0_,
                          alpha_i_=alpha_i_, omega_pi_=omega_pi_))


    ky_ = 11
    kz_ = np.sqrt((omega_0 ** 2) / (1 - omega_0 ** 2)) * ky_
    sol_approx = scipy.optimize.newton(disp_k_approx(ky=ky_, kz=kz_), omega_0 + 1e-3j, tol=1e-15)
    print(sol_approx)
    print("omega_k + i gamma = ", sol_approx)
    print("dispersion residual approx = ", abs(disp_k_approx(ky=ky_, kz=kz_)(sol_approx)))

    sol_full = scipy.optimize.newton(disp_k_full(ky=ky_, kz=kz_), sol_approx, tol=1e-15, maxiter=100)
    print(sol_full)
    print("omega_k + i gamma = ", sol_full)
    print("dispersion residual full = ", abs(disp_k_full(ky=ky_, kz=kz_)(sol_full)))

    ky_ = np.linspace(5, 60, 50)
    kz_ = np.sqrt((omega_0 ** 2) / (1 - omega_0 ** 2)) * ky_
    sol_approx_ = np.zeros((len(ky_)), dtype="complex128")
    sol_full_ = np.zeros((len(ky_)), dtype="complex128")
    k_abs = np.zeros((len(ky_)))

    for ii in range(len(ky_)):
        k_abs[ii] = np.sqrt(ky_[ii] ** 2 + kz_[ii] ** 2)
        try:
            sol_approx_[ii] = scipy.optimize.newton(disp_k_approx(ky=ky_[ii], kz=kz_[ii]), omega_0 + 1e-3j, tol=1e-15)
            sol_full_[ii] = scipy.optimize.newton(disp_k_full(ky=ky_[ii], kz=kz_[ii]), sol_approx_[ii], tol=1e-15)

        except:
            print("dispersion residual approx = ", abs(disp_k_approx(ky=ky_[ii], kz=kz_[ii])(sol_approx_[ii])))
            print("dispersion residual full = ", abs(disp_k_full(ky=ky_[ii], kz=kz_[ii])(sol_full_[ii])))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(k_abs, sol_approx_.real, linewidth=3, color="black")
    ax.plot(k_abs, sol_full_.real, linewidth=3, color="red")
    ax.set_ylabel(r'$\frac{\omega_r}{|\Omega_{ce}|}$', fontsize=22, labelpad=20, rotation=0)
    ax.set_xlabel(r"$|\vec{k}|d_{e}$", fontsize=15)
    ax.set_ylim(0, 0.7)
    ax.set_xlim(10, 70)
    ax.set_ylim(0.4, 0.6)
    ax.set_yticks([0.4, 0.5, 0.6])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(k_abs, sol_full_.imag, color="black", linewidth=3)
    ax.plot(k_abs, sol_approx_.imag, color="red", linewidth=3)
    ax.set_ylabel(r'$\frac{\gamma}{|\Omega_{ce}|}$', fontsize=22, labelpad=25, rotation=0)
    ax.set_xlabel(r"$|\vec{k}|d_{e}$", fontsize=15)
    ax.set_ylim(-0.0005, 0.012)
    ax.set_yticks([0, 0.005, 0.01])
    ax.set_yticklabels([0, 0.005, 0.01])
    ax.set_xlim(10, 70)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
