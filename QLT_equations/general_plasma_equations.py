"""Module basic plasma functions (e.g. Bessel and Z function)

Last modified: May 25th, 2025

Author: Opal Issan (oissan@ucsd.edu)
"""

import numpy as np
from scipy.special import wofz
import scipy



def Z(z):
    """plasma dispersion function Z(z)

    :param z: phase velocity argument
    :return: Z(z)
    """
    return 1j * np.sqrt(np.pi) * wofz(z)


def Z_prime(z):
    """derivative of the plasma dispersion function Z'(z)

    :param z: phase velocity argument
    :return: Z'(z)
    """
    return -2 * (1 + z * Z(z))


def I(Lambda, m):
    """modified Bessel function of the first kind I_{m}(Lambda) x exp(-Lambda)

    :param Lambda: argument of the Bessel function
    :param m: order of the modified Bessel function
    :return:I_{m}(Lambda) x exp(-Lambda)
    """
    return scipy.special.ive(m, Lambda)


def J(Lambda, m):
    """Bessel function of the first kind J_{m}(Lambda)

    :param Lambda: argument of the Bessel function
    :param m: order of the Bessel function
    :return: J_{m}(Lambda)
    """
    return scipy.special.jv(m, Lambda)

def trap(vec, dx):
    """trapezoidal rule for integration

    :param vec: 1a array, arbitrary vector
    :param dx: float, assume uniform spacing of the samples
    :return: trapezoidal approximation of the integral of the vector => int v(x) dx
    """
    return np.sum((vec[:-1] + vec[1:]) * 0.5 * dx)