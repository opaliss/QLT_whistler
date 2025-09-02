# ---------------------------------------------
# Solves electromagnetic dispersion for parallel
# propagation of EM waves
#
# Reference, Stix, Waves in Plasmas, Section 11-2
# (Propagation parallel to B0)
#
# See also https://farside.ph.utexas.edu/teaching/plasma/lectures1/node90.html
#
#
# Vadim Roytershteyn, 2021
# ---------------------------------------------
import numpy as np
from scipy.special import wofz
from scipy.optimize import root
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

class em_solver:
    sqrt_pi = np.sqrt(np.pi)
    sqrt_2 = np.sqrt(2.0)

    # initialization
    def __init__(self, P, nsp, ns, qs, ms, Wcs, Tpar, Tper, U0):
        # polarization
        self.P = P

        # T
        self.Tpar = Tpar
        self.Tper = Tper
        self.A = Tper / Tpar - 1

        # U0
        self.U0 = U0

        # density
        self.ns = ns

        # nsp
        self.nsp = nsp

        # qs
        self.qs = qs

        # ms
        self.ms = ms

        # Wcs
        self.Wcs = Wcs

        # thermal
        self.vth = np.sqrt(2 * Tpar / ms)

    # ---------------------------
    # plasma dispersion function
    # ---------------------------
    def Z(self, xi):
        Z_ = 1j * self.sqrt_pi * wofz(xi)
        return Z_

    # ---------------------------
    # Equation to solve
    # ---------------------------
    def epsilon(self, w, k):
        w_ = w[0] + 1j * w[1]
        eps_ = 1.0 - k ** 2 / w_ ** 2 + 0.0j

        for s in range(self.nsp):
            f_ = self.ns[s] * self.qs[s] ** 2 / self.ms[s] / w_ ** 2

            l1_ = (w_ - k * self.U0[s] + self.P * self.Wcs[s]) / (k * self.vth[s])
            l2_ = l1_ * self.Tper[s] / self.Tpar[s] - self.P * self.Wcs[s] / (k * self.vth[s])

            eps_ += f_ * (self.A[s] + l2_ * self.Z(l1_))

        return np.array([eps_.real, eps_.imag])

    # ---------------------------------------
    # Nonlinear solver: for a given k, return
    # the solution w
    # inputs : value of k and guess wg
    # ---------------------------------------
    def solve(self, k, wg):
        # initial guess
        wg_ = [wg.real, wg.imag]
        sol = root(self.epsilon, wg_, args=(k), tol=1E-15)
        res_ = sol.x[0] + 1j * sol.x[1]

        return (res_, sol.success)


#### main section   (whistler)

mime = 1836
wpewce = 4
nsp = 3  # number of species

Te = 0.00441942 ** 2  # reference electron temperature (cold)
Wci = +1 / wpewce / mime  # ion cyclotron / wpe
Wce = -1 / wpewce  # electron cyclotron / wpe
P = +1

Tpar_Te = np.array([1, 200, 1])  # ratio of temperatures to the electron
A = np.array([0, 4, 0])  # anisotropy parameter A = Tper/Tpar - 1

# everything is normalized to inertial length and plasma frequencies of the first species

ms = np.array([1, 1, mime])
qs = np.array([1, 1, -1])
ns = np.array([0.8, 0.2, 1.0])
Wcs = qs / ms / wpewce

Tpar = Te * Tpar_Te
Tper = Tpar * (A + 1)

U0 = np.array([0, 0, 0])

# create solver object

solver = em_solver(P=P, ns=ns, nsp=nsp, qs=qs, ms=ms, Tpar=Tpar, Tper=Tper, U0=U0, Wcs=Wcs)

# trace the dispersion relation

k = np.arange(2, 0.01, -0.01)
w = np.zeros(k.shape)
g = np.zeros(k.shape)

wg = 0.8 * Wce - 1E-4j

for j, k_ in enumerate(k):
    w_, err_ = solver.solve(k_, wg)
    wg = w_
    w[j] = w_.real / np.abs(Wce)
    g[j] = w_.imag / np.abs(Wce)

f, ax = plt.subplots(2, 1, sharex=True)

axc = ax[0]
axc.plot(k, w)
axc.set_ylabel(r'$\omega/\Omega_{ce}$')

axc = ax[1]
axc.plot(k, g)
axc.set_ylabel(r'$\gamma/\Omega_{ce}$')
axc.set_xlabel(r'${k}d_e$')

f.savefig('disp.pdf', bbox_inches='tight')

dat = np.stack((k, w, g), axis=1)

np.savetxt("disp_parallel.dat", dat)

plt.show()