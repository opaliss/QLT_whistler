"""Module with QLT equations describing the oblique electrostatic
secondary instability (oblique whistler)

References
----------
V. Roytershteyn and G. L. Delzanno.
Nonlinear coupling of whistler waves to oblique electrostatic turbulence enabled by cold plasma.
Physics of Plasmas, 28(4):042903, 04 2021

Last modified: May 25th, 2025

Author: Opal Issan (oissan@ucsd.edu)
"""

import numpy as np
from general_plasma_equations import I, J, Z_prime, Z


def sum_bessel(lambda_, omega, k_par, n_max, alpha_c_par):
    """sum of Bessel functions

    :param lambda_: float or 1d array, argument of the modified Bessel function
    :param omega: float or 1d array, frequency
    :param k_par: float or 1d array, parallel wavenumber
    :param n_max: int, maximum number of bessel function to include in the infinite sum
    :param alpha_c_par: float, sqrt(2T_{\| c}/m_{e})
    :return: sum of Bessel functions
    """
    sol = 0
    for n in range(-n_max, n_max + 1):
        xi = (omega + n) / (k_par * alpha_c_par)
        sol += I(m=n, Lambda=lambda_) * Z(z=xi)
    return sol


def electron_response(n_c, omega_pe, alpha_c_par, omega, k_par, n_max, k_perp):
    """linear electron response

    :param n_c: float, ratio of cold plasma density to total electron density
    :param omega_pe: float, electron plasma frequency
    :param alpha_c_par: float, sqrt(2T_{\| c}/m_{e})
    :param omega: float or 1d array, frequency
    :param k_par: float or 1d array, parallel wavenumber
    :param n_max: int, maximum number of bessel function to include in the infinite sum
    :param k_perp: float or 1d array, perpendicular wavenumber
    :return: linear electron response
    """
    # Bessel argument electron
    lambda_ = 0.5 * ((k_perp * alpha_c_par) ** 2)
    return 2 * n_c * ((omega_pe ** 2) / (alpha_c_par ** 2)) * (
                1 + (omega / (k_par * alpha_c_par))
                * sum_bessel(lambda_=lambda_, omega=omega, k_par=k_par, n_max=n_max, alpha_c_par=alpha_c_par))


def ion_response(omega_pi, alpha_i, m_star, k_perp, v_0, omega_0, omega, k_par):
    """linear ion response

    :param omega_pi: float, ion plasma frequency
    :param alpha_i: float, sqrt(2T_{i}/m_{i})
    :param m_star: int, most important bessel combination
    :param k_perp: float or 1d array, perpendicular wavenumber
    :param v_0: float, cold electron drift magnitude caused by the polarized electric field of the primary wave
    :param omega_0: float, frequency of the primary wave at saturation
    :param omega: float or 1d array, frequency
    :param k_perp: float or 1d array, parallel wavenumber
    :return: linear ion response
    """
    k_abs = np.sqrt(k_perp ** 2 + k_par ** 2)
    # Bessel argument ion Doppler-shifted
    a = k_perp * np.abs(v_0) / omega_0
    return (omega_pi ** 2) / (alpha_i ** 2) * (J(m=m_star, Lambda=a) ** 2) \
           * Z_prime(z=(omega - omega_0) / (alpha_i * k_abs))


def dispersion_relation(k_perp, k_par, omega_pe, omega_pi, omega_0, v_0, alpha_i,
                        alpha_c_par, n_c, m_star=-1, n_max=20):
    """dispersion relation of oblique electrostatic waves

    :param k_perp: float or 1d array, perpendicular wavenumber
    :param k_par: float or 1d array, parallel wavenumber
    :param omega_pe: float, electron plasma frequency
    :param omega_pi: float, ion plasma frequency
    :param omega_0: float, primary whistler wave frequency at saturation
    :param v_0: float, cold electron drift magnitude caused by the primary polarized electric
    :param alpha_i: float, sqrt(2T_{i}/m_{i})
    :param alpha_c_par: float, sqrt(2T_{\| c}/m_{e})
    :param n_c: float, ration of cold electron density over the total electron density
    :param m_star: int, most dominant contribution of bessel function for ion response
    :param n_max: int, maximum number fo terms to approximate the infinite sum in electron response
    :return: D(omega, k_perp, k_par)
    """
    return lambda omega: k_perp ** 2 + k_par ** 2 \
                         + electron_response(n_c=n_c, omega_pe=omega_pe, alpha_c_par=alpha_c_par,
                                             omega=omega, k_par=k_par, n_max=n_max, k_perp=k_perp) \
                         - ion_response(omega_pi=omega_pi, alpha_i=alpha_i, m_star=m_star, k_perp=k_perp, v_0=v_0,
                                        omega_0=omega_0, omega=omega, k_par=k_par)