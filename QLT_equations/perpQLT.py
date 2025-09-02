"""Module with QLT equations describing the quasi-perp
(ECDI-like) electrostatic secondary drift-driven instability

References
----------
V. Roytershteyn and G. L. Delzanno.
Nonlinear coupling of whistler waves to oblique electrostatic turbulence enabled by cold plasma.
Physics of Plasmas, 28(4):042903, 04 2021

Last modified: Sept 2nd, 2025

Author: Opal Issan (oissan@ucsd.edu)
"""

import numpy as np
from QLT_equations.general_plasma_equations import Z_prime, I, J


def dKdt(omega_pi, alpha_i, E_vec, k_vec, omega_vec, dk, omega_0, v_0, m_star=-3):
    """evolution equation for the perp kinetic energy of the cold electron population

    :param omega_pi: float, ion plasma frequency
    :param alpha_i: float, sqrt(2)v_{thi} proportional to the ion thermal speed
    :param E_vec: 1D array, vector with |E(k, t)|^2 entries
    :param k_vec: 1D array, vector of all relevant wavenumbers
    :param omega_vec: 1D array, frequency of all relevant wavenumbers
    :param dk: float, spacing in wavenumber coordinate
    :param omega_0: float, frequency of the primary driver whistler wave
    :param v_0: float, drift caused by the primary whistler wave
    :param m_star: int, most relevant sideband in the dispersion relation
    :return: dK_perp/dt
    """
    # initialize solution
    sol = np.zeros(len(k_vec))
    # loop over each wavenumber
    for ii in range(len(k_vec)):
        ions = ion_response(omega_pi=omega_pi, alpha_i=alpha_i, m_star=m_star,
                            omega=omega_vec[ii], omega_0=omega_0,
                            k_perp=k_vec[ii], v_0=v_0)
        sol[ii] = E_vec[ii] * (omega_vec[ii] * (1 - ions / (k_vec[ii] ** 2))).imag * k_vec[ii]
    # integrate over all relevant wavenumbers
    return - 1 / 4 / np.pi * np.sum(sol) * dk


def dEdt(gamma, E_vec):
    """electric field power spectrum d|E(k, t)|^2dt

    :param gamma: 1D array, gamma(k) growth rates
    :param E_vec: 1D array, |E(k, t)|^2 for the relevant growth rates
    :return: d|E(k, t)|^2dt
    """
    return 2 * gamma * E_vec


def dBdt(omega_pi, alpha_i, E_vec, k_vec, omega_vec, dk, omega_0, v_0, k_0):
    """magnetic field power spectrum d|B(k, t)|^2dt

    :param omega_pi: float, ion plasma frequency
    :param alpha_i: float, sqrt(2)v_{thi} proportional to the ion thermal speed
    :param E_vec: 1D array, vector with |E(k, t)|^2 entries
    :param k_vec: 1D array, vector of all relevant wavenumbers
    :param omega_vec: 1D array, frequency of all relevant wavenumbers
    :param dk: float, spacing in wavenumber coordinate
    :param omega_0: float, frequency of the primary driver whistler wave
    :param v_0: float, drift caused by the primary whistler wave
    :param k_0: float, whistler wave wavenumber
    :return: int d|B(k, t)|^2dt
    """
    # kinetic energy
    dK_perp_dt = dKdt(omega_pi=omega_pi, alpha_i=alpha_i,
                      E_vec=E_vec, k_vec=k_vec, omega_vec=omega_vec, dk=dk,
                      omega_0=omega_0, v_0=v_0)
    # potential electrostatic energy
    dE_dt = np.sum(dEdt(gamma=omega_vec.imag, E_vec=E_vec) * k_vec) * dk
    # constant related to change of coordinates
    const = 1 + (omega_0 / k_0 / np.abs(omega_0 - 1)) ** 2
    return - 8 * np.pi / const * (dK_perp_dt + 1 / 8 / np.pi * dE_dt)


def dVdt(omega_0, k_0, omega_pi, alpha_i, E_vec, k_vec, omega_vec, dk, v_0):
    """change in cold electron drift dVdt

    :param omega_pi: float, ion plasma frequency
    :param alpha_i: float, sqrt(2)v_{thi} proportional to the ion thermal speed
    :param E_vec: 1D array, vector with |E(k, t)|^2 entries
    :param k_vec: 1D array, vector of all relevant wavenumbers
    :param omega_vec: 1D array, frequency of all relevant wavenumbers
    :param dk: float, spacing in wavenumber coordinate
    :param omega_0: float, frequency of the primary driver whistler wave
    :param v_0: float, drift caused by the primary whistler wave
    :param k_0: float, wavenumber of the primary wave
    :return: dVdt
    """
    const = 1 / 4 / np.pi * ((omega_0 / k_0 / (omega_0 - 1)) ** 2)
    return const * dBdt(omega_pi=omega_pi, alpha_i=alpha_i, E_vec=E_vec, k_vec=k_vec,
                        omega_vec=omega_vec, dk=dk, omega_0=omega_0, v_0=v_0, k_0=k_0)


def sum_bessel(lambda_, omega, n_max=20):
    """Bessel function sum

    :param lambda_: float, argument of the Bessel function
    :param omega: float, frequency omega_r + i gamma
    :param n_max: int, maximum number of terms in the summation, default is 20
    :return: Bessel function sum
    """
    sol = 0
    for n in range(1, n_max):
        sol += I(m=n, Lambda=lambda_) * (n ** 2) / (omega ** 2 - n ** 2)
    return sol


def ion_response(omega_pi, alpha_i, m_star, omega, omega_0, k_perp, v_0):
    """linear ion response function

    :param omega_pi: float, ion plasma frequency
    :param alpha_i: float, sqrt(2)v_{thi} proportional to the ion thermal speed
    :param m_star: int, most relevant sideband in the dispersion relation
    :param omega: float or 1D array, frequency omega_r + i gamma
    :param omega_0: float, primary wave frequency
    :param k_perp: float or 1D array, wavenumber
    :param v_0: float, drift amplitude caused by the whistler wave
    :return: linear ion response
    """
    a = k_perp * np.abs(v_0) / omega_0
    return (omega_pi ** 2) / (alpha_i ** 2) * (J(m=m_star, Lambda=a) ** 2) \
           * Z_prime((omega + m_star * omega_0) / (np.abs(k_perp) * alpha_i))


def cold_electron_response(k_perp, omega, n_max, omega_pe, alpha_perp_c, n_c):
    """linear cold electron response

    :param k_perp: float or 1d array, wavenumber
    :param omega: float or 1d array, frequency
    :param n_max: int, maximum number of bessel terms to include in the summation
    :param omega_pe: float, electron plasma frequency
    :param alpha_perp_c: float, sqrt(2T_{perp c}/m_{e})  proportional to the ion thermal speed
    :param n_c: float, ratio of cold electron density to total cold electron density
    :return: linear cold electron response
    """
    lambda_ = (k_perp * alpha_perp_c / np.sqrt(2)) ** 2
    return -n_c * 4 * (omega_pe ** 2) / (alpha_perp_c ** 2) * sum_bessel(lambda_=lambda_, omega=omega, n_max=n_max)


def dispersion_relation(k_perp, omega_pe, omega_0, omega_pi, v_0, alpha_i, alpha_perp_c, n_c, m_star=-3, n_max=20):
    """linear dispersion relation of quasi-perp electrostatic secondary-instabilities

    :param k_perp: float or 1d array, wavenumber
    :param omega_pi: float, ion plasma frequency
    :param omega_pe: float, electron plasma frequency
    :param alpha_i: float, sqrt(2)v_{thi} proportional to the ion thermal speed
    :param omega_0: float, frequency of the primary driver whistler wave
    :param v_0: float, drift caused by the primary whistler wave
    :param alpha_perp_c: float, sqrt(2T_{perp c}/m_{e})  proportional to the ion thermal speed
    :param n_c: float, ratio of cold electron density to total cold electron density
    :param m_star: int, most relevant sideband in the dispersion relation
    :param n_max: int, maximum number of bessel terms to include in the summation, default is 20
    :return: D(omega, k_perp)
    """
    return lambda omega: k_perp ** 2 \
                         + cold_electron_response(k_perp=k_perp, omega=omega, n_max=n_max, omega_pe=omega_pe,
                                                  alpha_perp_c=alpha_perp_c, n_c=n_c) \
                         - ion_response(omega_pi=omega_pi, alpha_i=alpha_i, m_star=m_star, v_0=v_0, omega=omega,
                                        omega_0=omega_0, k_perp=k_perp)
