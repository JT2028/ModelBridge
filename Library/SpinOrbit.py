"""
Atomic spin-orbit coupling, lambda L.S on every atom (Fang 2015, Eq. 11-12 and Table VIII), in k space and real space.

Both functions take a spinless Hamiltonian and return the one with spin, twice the size, in spin-major order:
index = spin * n_spinless + spinless index, with spin 0 = up. spin_conserving=True keeps only Lz Sz, so the two
spins decouple.
"""

import numpy as np
from .Param import N_ORBITALS

SOC_STRENGTH = {"Mo": 0.0836, "W": 0.2874, "S": 0.0556, "Se": 0.2470}    # eV, Fang 2015 Table VIII

# ModelBridge orbital order: A = (dxz, dyz), B = (px, py, pz) odd, C = (dxy, dx2-y2, dz2), D = (px, py, pz) even
METAL_D = [0, 1, 5, 6, 7]                   # dxz, dyz, dxy, dx2-y2, dz2
CHALCOGEN_P = [[2, 3, 10], [8, 9, 4]]       # (px, py, pz): the odd pair with the even pz, the even pair with the odd pz

SQRT3 = np.sqrt(3)

# https://ar5iv.labs.arxiv.org/html/1808.05546
# <row| L |col> for the real orbitals (px, py, pz)
L_P = 1j * np.array([
    [[ 0,  0,  0],
     [ 0,  0, -1],
     [ 0,  1,  0]],

    [[ 0,  0,  1],
     [ 0,  0,  0],
     [-1,  0,  0]],

    [[ 0, -1,  0],
     [ 1,  0,  0],
     [ 0,  0,  0]],
])

# <row| L |col> for the real orbitals (dxz, dyz, dxy, dx2-y2, dz2)
# Jo et al. dz2 dx2-y2 dxy dyz dxz
L_D = 1j * np.array([
    [[ 0,      0,  1,  0,      0],
     [ 0,      0,  0, -1, -SQRT3],
     [-1,      0,  0,  0,      0],
     [ 0,      1,  0,  0,      0],
     [ 0,  SQRT3,  0,  0,      0]],

    [[ 0,      0,  0, -1,  SQRT3],
     [ 0,      0, -1,  0,      0],
     [ 0,      1,  0,  0,      0],
     [ 1,      0,  0,  0,      0],
     [-SQRT3,  0,  0,  0,      0]],

    [[ 0, -1,  0,  0,  0],
     [ 1,  0,  0,  0,  0],
     [ 0,  0,  0,  2,  0],
     [ 0,  0, -2,  0,  0],
     [ 0,  0,  0,  0,  0]],
])
# Pauli matrices for spin-1/2 
# sigma x sigma y
PAULI = np.array([
    [[0,   1], [1,  0]],
    [[0, -1j], [1j, 0]],
    [[1,   0], [0, -1]],
])


def angular_momentum():
    # 11x11 block for angular momentum block
    L = np.zeros((3, N_ORBITALS, N_ORBITALS), dtype=complex)

    L[np.ix_(range(3), METAL_D, METAL_D)] = L_D

    for shell in CHALCOGEN_P:
        L[np.ix_(range(3), shell, shell)] = L_P

    return L


def spin_orbit_onsite(lambda_metal, lambda_chalcogen, spin_conserving=False):
    L = angular_momentum()
    
    # 11x1 vecter [lambda_metal lambda_metal lambda_chalcogen lambda_chalcogen ...]
    strength = np.full(N_ORBITALS, lambda_chalcogen, dtype=float)
    strength[METAL_D] = lambda_metal
    
    # L·S = Lx Sx + Ly Sy + Lz Sz
    # spin-conserving components (only Lz Sz) or all components (Lx Sx + Ly Sy + Lz Sz)
    components = [2] if spin_conserving else [0, 1, 2]

    H_soc = np.zeros((2 * N_ORBITALS, 2 * N_ORBITALS), dtype=complex)

    for i in components:
        # S_i = 0.5 * sigma_i
        H_soc += 0.5 * np.kron(PAULI[i], strength[:, None] * L[i])

    return H_soc


'''
================================================
Below is for k space
================================================
'''
def add_spin_orbit(H_k, lambda_metal, lambda_chalcogen, spin_conserving=False):
    if H_k.shape[-2:] != (N_ORBITALS, N_ORBITALS):
        raise ValueError(f"H_k has shape {H_k.shape}; expected (n_k, {N_ORBITALS}, {N_ORBITALS}).")

    H_soc = spin_orbit_onsite(lambda_metal, lambda_chalcogen, spin_conserving)

    return np.kron(np.eye(2), H_k) + H_soc[None, :, :]


'''
================================================
Below is for real space
================================================
'''
def add_spin_orbit_real_space(H, lambda_metal, lambda_chalcogen, spin_conserving=False):
    n_spinless = H.shape[0]
    n_cells = n_spinless // N_ORBITALS

    if H.shape != (n_cells * N_ORBITALS, n_cells * N_ORBITALS):
        raise ValueError(f"H has shape {H.shape}; expected ({N_ORBITALS} * n_cells, {N_ORBITALS} * n_cells).")

    H_soc = spin_orbit_onsite(lambda_metal, lambda_chalcogen, spin_conserving)

    H_full = np.kron(np.eye(2), H)

    for cell in range(n_cells):
        # orbitals = indices of the orbitals for the current cell spin up
        orbitals = cell * N_ORBITALS + np.arange(N_ORBITALS)
        # then the same cell, spin down
        spin_orbitals = np.concatenate([orbitals, n_spinless + orbitals])

        H_full[np.ix_(spin_orbitals, spin_orbitals)] += H_soc

    return H_full
