"""Module with functions to solve the full-dispersion relation for oblique waves

References
----------
V. Roytershteyn and G. L. Delzanno.
Nonlinear coupling of whistler waves to oblique electrostatic turbulence enabled by cold plasma.
Physics of Plasmas, 28(4):042903, 04 2021

Last modified: August 25th, 2025

Author: Opal Issan (oissan@ucsd.edu)
"""
from QLT_equations.general_plasma_equations import Z_prime, J
from QLT_equations.obliqueQLT import electron_response
import numpy as np


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


def D_matrix(omega, ky, kz, n_c_, omega_pe_, alpha_c_par_,
             alpha_c_perp_, omega_0_, v_0_, alpha_i_, omega_pi_, N=5):
    # initialize matrix
    D_mat = np.zeros((N * 2 + 1, N * 2 + 1), dtype="complex128")
    k2 = ky ** 2 + kz ** 2
    for n in range(-N, N + 1):
        i1 = n + N
        # cold electron components
        D_mat[i1, i1] += k2 + electron_response(n_c=n_c_, omega_pe=omega_pe_,
                                                alpha_c_par=alpha_c_par_,
                                                alpha_c_perp=alpha_c_perp_, omega=omega + n * omega_0_,
                                                k_par=kz, k_perp=ky, n_max=20)

        for M in range(-N, N + 1):
            i2 = M + N
            D_mat[i1, i2] += -THETA(omega_pi_=omega_pi_, alpha_i_=alpha_i_, n=n, M=M, ky=ky,
                                    v_0_=v_0_, omega_0_=omega_0_, omega=omega, kz=kz)
    return D_mat
