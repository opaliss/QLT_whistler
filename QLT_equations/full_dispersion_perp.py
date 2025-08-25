"""Module with functions to solve the full-dispersion relation for perpendicular waves

References
----------
V. Roytershteyn and G. L. Delzanno.
Nonlinear coupling of whistler waves to oblique electrostatic turbulence enabled by cold plasma.
Physics of Plasmas, 28(4):042903, 04 2021

Last modified: August 25th, 2025

Author: Opal Issan (oissan@ucsd.edu)
"""
import numpy as np
from QLT_equations.general_plasma_equations import Z_prime, J
from QLT_equations.perpQLT import cold_electron_response


def THETA(omega_pi_, alpha_i_, n, M, k_perp, v_0_, omega_0_, omega, m_max=20):
    # magnitude of the wavevector
    k_abs = np.abs(k_perp)
    # Bessel argument ion Doppler-shifted
    a = k_perp * np.abs(v_0_) / omega_0_
    res = 0
    for m in range(-m_max, m_max + 1):
        res += (omega_pi_ ** 2) / (alpha_i_ ** 2) * J(m=m - n, Lambda=-a) * J(m=m - M, Lambda=-a) \
               * Z_prime(z=(omega + m * omega_0_) / (alpha_i_ * k_abs))
    return res


def D_matrix(omega, k_perp, n_c_, omega_pe_, alpha_c_perp_, omega_0_, v_0_, alpha_i_, omega_pi_, N=5):
    # initialize matrix
    D_mat = np.zeros((N * 2 + 1, N * 2 + 1), dtype="complex128")
    for n in range(-N, N + 1):
        i1 = n + N
        # cold electron components
        D_mat[i1, i1] += k_perp ** 2 + cold_electron_response(k_perp=k_perp, omega=omega + n * omega_0_,
                                                              n_max=20, omega_pe=omega_pe_, alpha_perp_c=alpha_c_perp_,
                                                              n_c=n_c_)

        for M in range(-N, N + 1):
            i2 = M + N
            D_mat[i1, i2] += -THETA(omega_pi_=omega_pi_, alpha_i_=alpha_i_, n=n, M=M, k_perp=k_perp,
                                    v_0_=v_0_, omega_0_=omega_0_, omega=omega)
    return D_mat