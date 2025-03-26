"""
(resonant) QLT functions

Author: Opal Issan (oissan@ucsd.edu)
Last modified: March 26th, 2025
"""
import numpy as np
from scipy.special import wofz
import scipy


def Z(xi):
    # plasma dispersion function
    return 1j * np.sqrt(np.pi) * wofz(xi)


def xi_1(z, q, beta_par, M=1, ions=False):
    # phase velocity (with cycltron frequency)
    if ions:
        return (M * z + 1) / (np.abs(q) * np.sqrt(beta_par) * np.sqrt(M))
    else:
        return (z - 1) / (np.abs(q) * np.sqrt(beta_par))


def xi_0(z, q, beta_par, M=1):
    # phase velocity
    return z / (np.abs(q) * np.sqrt(beta_par) * np.sqrt(M))


def eta(z, A, q, beta_par):
    # non-dimensional number
    return (z * (A + 1) - A) / (np.abs(q) * np.sqrt(beta_par))


# dispersion relation
def dispersion_relation(q, beta_par_c, beta_par_h, beta_par_i, A_c, A_h, delta, M=1836, include_cold=True):
    if include_cold:
        return lambda z: q ** 2 - xi_0(z=z, q=q, beta_par=beta_par_i, M=M) * Z(
            xi=xi_1(z=z, q=q, beta_par=beta_par_i, M=M)) \
                         - (1 - delta) * (A_c + eta(z, A=A_c, q=q, beta_par=beta_par_c) * Z(
            xi=xi_1(z, q=q, beta_par=beta_par_c))) \
                         - delta * (A_h + eta(z, A=A_h, q=q, beta_par=beta_par_h) * Z(
            xi=xi_1(z, q=q, beta_par=beta_par_h)))
    else:
        return lambda z: q ** 2 - xi_0(z=z, q=q, beta_par=beta_par_i, M=M) * Z(
            xi=xi_1(z=z, q=q, beta_par=beta_par_i, M=M)) - delta * (A_h + eta(z, A=A_h, q=q, beta_par=beta_par_h) * Z(
            xi=xi_1(z, q=q, beta_par=beta_par_h)))


def get_z_vec(q_vec, A_h, A_c, beta_par_c, beta_par_h, beta_par_i, delta, include_cold=True):
    z_vec = np.zeros(len(q_vec), dtype="complex128")
    for ii, q in enumerate(q_vec):
        if q < 1.5:
            ic1 = 0.55 + 1e-4j
            ic2 = 0.8 + 1e-4j
        else:
            ic1 = 0.65 + 1e-4j
            ic2 = 0.8 + 1e-4j
        try:
            z_vec[ii] = scipy.optimize.newton(
                dispersion_relation(q=q, A_c=A_c, beta_par_c=beta_par_c, A_h=A_h, beta_par_h=beta_par_h,
                                    beta_par_i=beta_par_i, delta=delta, include_cold=include_cold), ic1, tol=1e-15)
            #if dispersion_relation(q=q, A_c=A_c, beta_par_c=beta_par_c, A_h=A_h, beta_par_h=beta_par_h,
             #                      beta_par_i=beta_par_i, delta=delta, include_cold=include_cold)(z_vec[ii]) > 1e-10:
                # print("q1=", q)
                # print("residual1=", np.abs(
                #     dispersion_relation(q=q, A_c=A_c, beta_par_c=beta_par_c, A_h=A_h, beta_par_h=beta_par_h,
                #                         beta_par_i=beta_par_i, delta=delta, include_cold=include_cold)(z_vec[ii])))
        except:
            try:
                z_vec[ii] = scipy.optimize.newton(
                    dispersion_relation(q=q, A_c=A_c, beta_par_c=beta_par_c, A_h=A_h, beta_par_h=beta_par_h,
                                        beta_par_i=beta_par_i, delta=delta, include_cold=include_cold), ic2, tol=1e-15)
                #if dispersion_relation(q=q, A_c=A_c, beta_par_c=beta_par_c, A_h=A_h, beta_par_h=beta_par_h,
                #                       beta_par_i=beta_par_i, delta=delta, include_cold=include_cold)(
                    # z_vec[ii]) > 1e-10:
                    # print("q3=", q)
                    # print("residual3=", np.abs(
                    #     dispersion_relation(q=q, A_c=A_c, beta_par_c=beta_par_c, A_h=A_h, beta_par_h=beta_par_h,
                    #                         beta_par_i=beta_par_i, delta=delta, include_cold=include_cold)(z_vec[ii])))
            except:
                z_vec[ii] = 0
    return z_vec


def dB_dt(gamma, B_vec):
    return 2 * gamma * B_vec


def dbetaperp_dt(A, q_vec, B_vec, z_vec, dq, beta_par):
    # Eq. (2) rhs
    rhs = np.zeros(len(q_vec))
    for ii in range(len(q_vec)):
        # phase velocity normalized
        xi_1_e = xi_1(z=z_vec[ii], q=q_vec[ii], beta_par=beta_par)
        xi_0_e = xi_0(z=z_vec[ii], q=q_vec[ii], beta_par=beta_par)
        # rhs
        rhs[ii] = (B_vec[ii] / (q_vec[ii] ** 2)) * (xi_0_e.real - xi_1_e.real) * np.exp(-(xi_1_e.real) ** 2) * (
                (z_vec[ii].real - 1) * A + z_vec[ii].real)
    return 2 * np.sqrt(np.pi) * np.sum((rhs[:-1] + rhs[1:]) * 0.5 * dq)  # trapezoidal rule


def dbetapar_dt(A, q_vec, B_vec, z_vec, dq, beta_par):
    # Eq. (3) rhs
    rhs = np.zeros(len(q_vec))
    for ii in range(len(q_vec)):
        # phase velocity normalized
        xi_1_e = xi_1(z=z_vec[ii], q=q_vec[ii], beta_par=beta_par)
        xi_0_e = xi_0(z=z_vec[ii], q=q_vec[ii], beta_par=beta_par)
        # rhs
        rhs[ii] = (B_vec[ii] / (q_vec[ii] ** 2)) * (xi_1_e.real) * np.exp(-(xi_1_e.real) ** 2) * (
                (z_vec[ii].real - 1) * A + z_vec[ii].real)
    return 4 * np.sqrt(np.pi) * np.sum((rhs[:-1] + rhs[1:]) * 0.5 * dq)  # trapezoidal rule


def dydt(t, f, q_vec, delta, beta_par_i, include_cold):
    """

    :param t:
    :param f:
    :param q_vec:
    :param delta:
    :param beta_par_i:
    :param include_cold:
    :return:
    """
    # anisotropy cold
    if include_cold:
        A_c = f[0] / f[1] - 1
    else:
        A_c = 0

    # anisotropy hot
    A_h = f[2] / f[3] - 1

    # dispersion solver
    z_vec = get_z_vec(q_vec, A_c=A_c, beta_par_c=f[1],
                      A_h=A_h, beta_par_h=f[3], delta=delta,
                      beta_par_i=beta_par_i, include_cold=include_cold)

    # cold
    # beta perpendicular
    if include_cold:
        rhs_beta_perp_c = dbetaperp_dt(A=A_c, q_vec=q_vec, B_vec=f[4:], z_vec=z_vec, dq=q_vec[1] - q_vec[0],
                                       beta_par=f[1])
        # beta parallel
        rhs_beta_par_c = dbetapar_dt(A=A_c, q_vec=q_vec, B_vec=f[4:], z_vec=z_vec, dq=q_vec[1] - q_vec[0],
                                     beta_par=f[1])
    else:
        rhs_beta_perp_c = 0
        rhs_beta_par_c = 0

        # hot
    # beta perpendicular
    rhs_beta_perp_h = dbetaperp_dt(A=A_h, q_vec=q_vec, B_vec=f[4:], z_vec=z_vec, dq=q_vec[1] - q_vec[0], beta_par=f[3])
    # beta parallel
    rhs_beta_par_h = dbetapar_dt(A=A_h, q_vec=q_vec, B_vec=f[4:], z_vec=z_vec, dq=q_vec[1] - q_vec[0], beta_par=f[3])

    # magnetic energy
    rhs_B = dB_dt(gamma=z_vec.imag, B_vec=f[4:])

    return np.concatenate(([rhs_beta_perp_c], [rhs_beta_par_c], [rhs_beta_perp_h], [rhs_beta_par_h], rhs_B))
