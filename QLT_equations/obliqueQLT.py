"""Module with QLT equations describing the oblique electrostatic
 secondary waves instability (oblique whistler)

References
----------
V. Roytershteyn and G. L. Delzanno.
Nonlinear coupling of whistler waves to oblique electrostatic turbulence enabled by cold plasma.
Physics of Plasmas, 28(4):042903, 04 2021.

Last modified: July 16th, 2025

Author: Opal Issan (oissan@ucsd.edu)
"""

import numpy as np
from QLT_equations.general_plasma_equations import I, J, Z_prime, Z
from QLT_equations.perpQLT import dEdt


def sum_bessel(lambda_, omega, k_par, alpha_c_par, n_max=20, n_factor=0, include_Z=True):
    """sum of Bessel functions

    :param lambda_: float or 1d array, argument of the modified Bessel function
    :param omega: float or 1d array, frequency
    :param k_par: float or 1d array, parallel wavenumber
    :param alpha_c_par: float, sqrt(2T_{\| c}/m_{e})
    :param n_max: int, maximum number of bessel function to include in the infinite sum, default is 20
    :param n_factor: int, multiply by n^{n_factor}, default is 0
    :param include_Z: bool, include or exclude Z function, default is True
    :return: sum of Bessel functions
    """
    sol = 0
    for n in range(-n_max, n_max + 1):
        xi = (omega - n) / (k_par * alpha_c_par)
        if include_Z:
            sol += I(m=n, Lambda=lambda_) * Z(z=xi) * (n ** n_factor)
        else:
            sol += I(m=n, Lambda=lambda_) * (n ** n_factor)
    return sol


def electron_response(n_c, omega_pe, alpha_c_par, alpha_c_perp, omega, k_par, n_max, k_perp):
    """linear electron response

    :param n_c: float, ratio of cold plasma density to total electron density
    :param omega_pe: float, electron plasma frequency
    :param alpha_c_par: float, sqrt(2T_{\| c}/m_{e})
    :param alpha_c_perp: float, sqrt(2T_{\perp c}/m_{e})
    :param omega: float or 1d array, frequency
    :param k_par: float or 1d array, parallel wavenumber
    :param n_max: int, maximum number of bessel function to include in the infinite sum
    :param k_perp: float or 1d array, perp wavenumber
    :return: linear electron response
    """
    # Bessel argument electron
    lambda_ = 0.5 * ((k_perp * alpha_c_perp) ** 2)
    anisotropy_const = (alpha_c_par / alpha_c_perp) ** 2 - 1
    return 2 * n_c * ((omega_pe ** 2) / (alpha_c_par ** 2)) * (
            1 + (omega / (k_par * alpha_c_par))
            * sum_bessel(lambda_=lambda_, omega=omega,
                         k_par=k_par, n_max=n_max, alpha_c_par=alpha_c_par, n_factor=0)
            + (1 / k_par / alpha_c_par * anisotropy_const)
            * sum_bessel(lambda_=lambda_, omega=omega,
                         k_par=k_par, n_max=n_max, alpha_c_par=alpha_c_par, n_factor=1))


def ion_response(omega_pi, alpha_i, k_perp, v_0, omega_0, omega, k_par, m_star=-1):
    """linear ion response

    :param omega_pi: float, ion plasma frequency
    :param alpha_i: float, sqrt(2T_{i}/m_{i})
    :param m_star: int, most important bessel combination, default is m_star=-1
    :param k_perp: float or 1d array, perp wavenumber
    :param k_par: float or 1d array, parallel wavenumber
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
           * Z_prime(z=(omega + m_star * omega_0) / (alpha_i * k_abs))


def dispersion_relation(k_perp, k_par, omega_pe, omega_pi, omega_0, v_0, alpha_i,
                        alpha_c_par, alpha_c_perp, n_c, m_star=-1, n_max=20):
    """dispersion relation of oblique electrostatic waves

    :param k_perp: float or 1d array, perp wavenumber
    :param k_par: float or 1d array, parallel wavenumber
    :param omega_pe: float, electron plasma frequency
    :param omega_pi: float, ion plasma frequency
    :param omega_0: float, primary whistler wave frequency at saturation
    :param v_0: float, cold electron drift magnitude caused by the primary polarized electric
    :param alpha_i: float, sqrt(2T_{i}/m_{i})
    :param alpha_c_par: float, sqrt(2T_{\| c}/m_{e})
    :param alpha_c_perp: float, sqrt(2T_{\perp c}/m_{e})
    :param n_c: float, ration of cold electron density over the total electron density
    :param m_star: int, most dominant contribution of bessel function for ion response, default is 50
    :param n_max: int, maximum number fo terms to approximate the infinite sum in electron response, default is -1
    :return: D(omega, k_perp, k_par)
    """
    return lambda omega: k_perp ** 2 + k_par ** 2 \
                         + electron_response(n_c=n_c, omega_pe=omega_pe, alpha_c_par=alpha_c_par,
                                             alpha_c_perp=alpha_c_perp, omega=omega, k_par=k_par,
                                             n_max=n_max, k_perp=k_perp) \
                         - ion_response(omega_pi=omega_pi, alpha_i=alpha_i, m_star=m_star, k_perp=k_perp, v_0=v_0,
                                        omega_0=omega_0, omega=omega, k_par=k_par)


def dKperpdt(E_vec, omega_pe, alpha_c_par, alpha_c_perp, n_c, k_par, k_perp, omega_vec, dk):
    """

    :param dk: float, spacing in wavenumber magnitude
    :param omega_vec: float or 1d array, frequency
    :param alpha_c_perp:  float, sqrt(2T_{\perp c}/m_{e})
    :param alpha_c_par: float, sqrt(2T_{\| c}/m_{e})
    :param E_vec: float or 1d array, psd of electric field
    :param omega_pe: float, plasma frequency
    :param n_c: float, cold plasma density
    :param k_perp: float or 1d array, perp wavenumber
    :param k_par: float or 1d array, parallel wavenumber
    :return: dKperpdt
    """
    # initialize solution
    sol = np.zeros(len(k_par))
    # anisotropy level
    anisotropy_term = (alpha_c_par / alpha_c_perp) ** 2 - 1
    # loop over all relevant k's
    for ii in range(len(k_par)):
        # wavenumber magnitude
        k2 = k_par[ii] ** 2 + k_perp[ii] ** 2
        # lambda parameter => bessel argument
        lambda_ = 0.5 * ((k_perp[ii] * alpha_c_perp) ** 2)
        # first term involving n^{1}
        term_1 = (omega_vec[ii] / (k_par[ii] * alpha_c_par)) * sum_bessel(lambda_=lambda_,
                                                                          omega=omega_vec[ii],
                                                                          k_par=k_par[ii],
                                                                          alpha_c_par=alpha_c_par,
                                                                          n_factor=1)
        # second term involving n^{2}
        term_2 = (1 / k_par[ii] / alpha_c_par * anisotropy_term) * sum_bessel(lambda_=lambda_,
                                                                              omega=omega_vec[ii],
                                                                              k_par=k_par[ii],
                                                                              alpha_c_par=alpha_c_par,
                                                                              n_factor=2)
        # final term (summing term_1 and term_2)
        sol[ii] = E_vec[ii] / k2 * (term_1 + term_2).imag * k_perp[ii]

    # return the integral over all k
    return 0.5 * n_c / np.pi * (omega_pe ** 2) / (alpha_c_par ** 2) * np.sum(sol) * dk


def dKpardt(E_vec, omega_pe, alpha_c_par, alpha_c_perp, n_c, k_par, k_perp, omega_vec, dk):
    """

    :param E_vec: float or 1d array, psd of electric field
    :param omega_pe: float, plasma frequency
    :param alpha_c_perp:  float, sqrt(2T_{\perp c}/m_{e})
    :param alpha_c_par: float, sqrt(2T_{\| c}/m_{e})
    :param n_c: float, cold plasma density
    :param dk: float, spacing in wavenumbers dk_perp*dk_par
    :param omega_vec: float or 1d array, frequency
    :param k_perp: float or 1d array, perp wavenumber
    :param k_par: float or 1d array, parallel wavenumber
    :return: dKpardt
    """
    # initialize solution
    sol = np.zeros(len(k_par))
    # anisotropy level
    anisotropy_term = (alpha_c_par / alpha_c_perp) ** 2 - 1
    # loop over all relevant k's
    for ii in range(len(k_par)):
        # wavenumber magnitude
        k2 = k_par[ii] ** 2 + k_perp[ii] ** 2
        # lambda parameter => bessel argument
        lambda_ = 0.5 * ((k_perp[ii] * alpha_c_perp) ** 2)
        # first term involving n^{0}
        term_1 = (omega_vec[ii] ** 2 / alpha_c_par / k_par[ii]) * sum_bessel(lambda_=lambda_,
                                                                             omega=omega_vec[ii],
                                                                             k_par=k_par[ii],
                                                                             alpha_c_par=alpha_c_par,
                                                                             n_factor=0)
        # second term involving n^{1}
        term_2 = (omega_vec[ii] * (anisotropy_term - 1) / k_par[ii] / alpha_c_par) * sum_bessel(lambda_=lambda_,
                                                                                                omega=omega_vec[ii],
                                                                                                k_par=k_par[ii],
                                                                                                alpha_c_par=alpha_c_par,
                                                                                                n_factor=1)
        # third term involving n^{2}
        term_3 = (-anisotropy_term / k_par[ii] / alpha_c_par) * sum_bessel(lambda_=lambda_,
                                                                           omega=omega_vec[ii],
                                                                           k_par=k_par[ii],
                                                                           alpha_c_par=alpha_c_par,
                                                                           n_factor=2)
        # summing all terms (term_1 + term_2 + term_3)
        sol[ii] = E_vec[ii] / k2 * (omega_vec[ii] + term_1 + term_2 + term_3).imag * k_perp[ii]

    # integration over all k's
    return n_c / np.pi * (omega_pe ** 2) / (alpha_c_par ** 2) * np.sum(sol) * dk


def dTperpdt(E_vec, omega_pe, alpha_c_par, alpha_c_perp, k_par, k_perp, omega_vec, dk, n_max=10):
    """

    :param n_max: n_max: int, maximum number of bessel function to include in the infinite sum (default is 10)
    :param E_vec: float or 1d array, psd of electric field
    :param omega_pe: float, plasma frequency
    :param alpha_c_perp:  float, sqrt(2T_{\perp c}/m_{e})
    :param alpha_c_par: float, sqrt(2T_{\| c}/m_{e})
    :param dk: float, spacing in wavenumbers dk_perp*dk_par
    :param omega_vec: float or 1d array, frequency
    :param k_perp: float or 1d array, perp wavenumber
    :param k_par: float or 1d array, parallel wavenumber
    :return: dTperpdt
    """
    # initialize solution
    sol = np.zeros(len(k_par))
    # anisotropy level
    anisotropy_term = (alpha_c_par / alpha_c_perp) ** 2 - 1
    # loop over k's
    for ii in range(len(k_par)):
        # k vector magnitude square
        k2 = k_par[ii] ** 2 + k_perp[ii] ** 2
        # lambda
        lambda_ = 0.5 * ((k_perp[ii] * alpha_c_perp) ** 2)
        # initialize the first term
        term_1 = 0
        for n in range(-n_max, n_max + 1):
            xi_n = ((omega_vec[ii] - n) / k_par[ii] / alpha_c_par).real
            xi_0 = (omega_vec[ii] / k_par[ii] / alpha_c_par).real
            exp = np.exp(-xi_n ** 2)
            add = n * I(Lambda=lambda_, m=n) * (xi_0 + n / k_par[ii] / alpha_c_par * anisotropy_term)
            term_1 += add * exp
        sol[ii] = E_vec[ii] / k2 * term_1 * k_perp[ii]
    # integrate over all k's
    return 0.5 * (omega_pe ** 2) / (alpha_c_par ** 2) / np.sqrt(np.pi) * np.sum(sol) * dk


def dTpardt(E_vec, omega_pe, alpha_c_par, alpha_c_perp, k_par, k_perp, omega_vec, dk, n_max=10):
    """

    :param n_max: n_max: int, maximum number of bessel function to include in the infinite sum (default is 10)
    :param E_vec: float or 1d array, psd of electric field
    :param omega_pe: float, plasma frequency
    :param alpha_c_perp:  float, sqrt(2T_{\perp c}/m_{e})
    :param alpha_c_par: float, sqrt(2T_{\| c}/m_{e})
    :param dk: float, spacing in wavenumbers dk_perp*dk_par
    :param omega_vec: float or 1d array, frequency
    :param k_perp: float or 1d array, perp wavenumber
    :param k_par: float or 1d array, parallel wavenumber
    :return: dTpardt
    """
    # initialize solution
    sol = np.zeros(len(k_par))
    # anisotropy level
    anisotropy_term = (alpha_c_par / alpha_c_perp) ** 2 - 1
    # loop over k's
    for ii in range(len(k_par)):
        # k vector magnitude squared
        k2 = k_perp[ii] ** 2 + k_par[ii] ** 2
        # lambda
        lambda_ = 0.5 * ((k_perp[ii] * alpha_c_perp) ** 2)
        # initialize the first term
        term_1 = 0
        for n in range(-n_max, n_max + 1):
            xi_n = ((omega_vec[ii] - n) / k_par[ii] / alpha_c_par).real
            exp = np.exp(-xi_n ** 2)
            term_1 += I(Lambda=lambda_, m=n) * (omega_vec[ii].real + n * anisotropy_term) * xi_n * exp
        sol[ii] = E_vec[ii] / k2 * term_1 * k_perp[ii]
    # integrate over all k's
    return (omega_pe ** 2) / (alpha_c_par ** 2) / np.sqrt(np.pi) * np.sum(sol) * dk


def dBdt(E_vec, omega_pe, alpha_c_par, alpha_c_perp, n_c, k_par, k_perp, omega_vec, dk, omega_0, k_0):
    """

    :param E_vec: float or 1d array, psd of electric field
    :param omega_pe: float, plasma frequency
    :param alpha_c_perp:  float, sqrt(2T_{\perp c}/m_{e})
    :param alpha_c_par: float, sqrt(2T_{\| c}/m_{e})
    :param dk: float, spacing in wavenumbers dk_perp*dk_par
    :param omega_vec: float or 1d array, frequency
    :param k_perp: float or 1d array, perp wavenumber
    :param k_par: float or 1d array, parallel wavenumber
    :param n_c: float, cold plasma density
    :param omega_0: float, whistler wave frequency
    :param k_0: float, whistler wave wavenumber
    :return: dBdt
    """
    dK_perp_dt = dKperpdt(E_vec=E_vec, omega_pe=omega_pe, alpha_c_par=alpha_c_par, alpha_c_perp=alpha_c_perp,
                          n_c=n_c, k_par=k_par, k_perp=k_perp, omega_vec=omega_vec, dk=dk)
    dK_par_dt = dKpardt(E_vec=E_vec, omega_pe=omega_pe, alpha_c_par=alpha_c_par, alpha_c_perp=alpha_c_perp,
                        n_c=n_c, k_par=k_par, k_perp=k_perp, omega_vec=omega_vec, dk=dk)
    dE_dt = np.sum(dEdt(gamma=omega_vec.imag, E_vec=E_vec) * k_perp) * dk
    const = 1 + (omega_0 / k_0 / (1 - omega_0))**2
    return - 8 * np.pi / const * (dK_perp_dt + 0.5 * dK_par_dt + 1 / 8 / np.pi * dE_dt)


def dVdt(omega_0, k_0, E_vec, omega_pe, alpha_c_par, alpha_c_perp, n_c, k_par, k_perp, omega_vec, dk):
    """

    :param omega_0: float, whistler wave frequency
    :param k_0: float, whistler wave wavenumber
    :param E_vec: float or 1d array, psd of electric field
    :param omega_pe: float, plasma frequency
    :param alpha_c_perp:  float, sqrt(2T_{\perp c}/m_{e})
    :param alpha_c_par: float, sqrt(2T_{\| c}/m_{e})
    :param n_c: float, cold plasma density
    :param dk: float, spacing in wavenumbers dk_perp*dk_par
    :param omega_vec: float or 1d array, frequency
    :param k_perp: float or 1d array, perp wavenumber
    :param k_par: float or 1d array, parallel wavenumber
    :param dk: float, spacing in wavenumbers dk_perp*dk_par
    :param omega_vec: float or 1d array, frequency
    :return: dVdt
    """
    const = 1 / 4 / np.pi * ((omega_0 / k_0 / (omega_0 - 1)) ** 2)
    return const * dBdt(E_vec=E_vec, omega_pe=omega_pe, alpha_c_par=alpha_c_par, alpha_c_perp=alpha_c_perp,
                        n_c=n_c, k_par=k_par, k_perp=k_perp, omega_vec=omega_vec, dk=dk, omega_0=omega_0, k_0=k_0)
