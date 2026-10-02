"""
Bloch Hamiltonian H(k) of the uniformly strained monolayer, built from the same blocks as build_full_H.

Momenta are fractional: k = k_frac[0] * b1 + k_frac[1] * b2, with a1 = a (1, 0), a2 = a (-1/2, sqrt(3)/2).
Cell gauge: a hop to the cell (n1, n2) carries exp(2 pi i k_frac . (n1, n2)), with no phase inside the cell.
K = (1/3, 1/3), K' = (2/3, 2/3), M = (1/2, 0). block_builder is build_H, or build_H_gw for the GW bands.
"""

import numpy as np
from .Hamiltonian import build_H
from .Neighbor import C3_half_neighbor_list
from .Param import GROUPS, GROUP_SIZE, N_ORBITALS, VALID

LATTICE_CONSTANT = 3.182    # Angstrom, MoS2

NO_STRAIN = {"exx": 0.0, "eyy": 0.0, "exy": 0.0}

HIGH_SYMMETRY = {
    "G": (0.0, 0.0),
    "M": (0.5, 0.0),
    "K": (1 / 3, 1 / 3),
    "K'": (2 / 3, 2 / 3),
}


def lattice_vectors(a=LATTICE_CONSTANT):
    lattice = a * np.array([
        [ 1.0,              0.0],
        [-0.5, np.sqrt(3) / 2.0],
    ])

    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T

    return lattice, reciprocal


def orbital_slices():
    slices = {}
    start = 0

    for group in GROUPS:
        slices[group] = slice(start, start + GROUP_SIZE[group])
        start += GROUP_SIZE[group]

    return slices


'''
================================================
Below is for the Bloch Hamiltonian
================================================
'''
def hopping_blocks(strain=None, block_builder=build_H):
    if strain is None:
        strain = NO_STRAIN

    slices = orbital_slices()
    blocks = []

    for group in GROUPS:
        instr = {"order": 0, "alpha": group, "beta": group, "c3": 0}

        blocks.append({
            "cell": (0, 0),
            "rows": slices[group],
            "cols": slices[group],
            "H": block_builder(instr, strain),
            "onsite": True,
        })

    for order, neighbors in zip([1, 2, 3], C3_half_neighbor_list(0, 0)):
        for tar, c3 in neighbors:
            for alpha, beta in sorted(VALID[order]):
                instr = {"order": order, "alpha": alpha, "beta": beta, "c3": c3}

                blocks.append({
                    "cell": tar,
                    "rows": slices[alpha],
                    "cols": slices[beta],
                    "H": block_builder(instr, strain),
                    "onsite": False,
                })

    return blocks


def bloch_hamiltonian(blocks, k_frac):
    k_frac = np.atleast_2d(k_frac)

    H_k = np.zeros((len(k_frac), N_ORBITALS, N_ORBITALS), dtype=complex)

    for block in blocks:
        cell = np.array(block["cell"], dtype=float)
        phase = np.exp(2j * np.pi * (k_frac @ cell))[:, None, None]

        H_k[:, block["rows"], block["cols"]] += phase * block["H"]

        if not block["onsite"]:
            H_k[:, block["cols"], block["rows"]] += np.conj(phase) * block["H"].conj().T

    return H_k


'''
================================================
Below is for momentum meshes and paths
================================================
'''
def mesh_labels(mesh_size):
    return np.indices((mesh_size, mesh_size)).reshape(2, -1).T


def uniform_mesh(mesh_size):
    return mesh_labels(mesh_size) / mesh_size


def k_path(point_names, points_per_segment, a=LATTICE_CONSTANT):
    _, reciprocal = lattice_vectors(a)

    corners = np.array([HIGH_SYMMETRY[name] for name in point_names])
    fractions = np.linspace(0.0, 1.0, points_per_segment, endpoint=False)[:, None]

    segments = []
    for start, end in zip(corners[:-1], corners[1:]):
        segments.append(start + fractions * (end - start))

    k_frac = np.vstack(segments + [corners[-1:]])

    steps = np.linalg.norm(np.diff(k_frac @ reciprocal, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(steps)])
    ticks = distance[::points_per_segment]

    return k_frac, distance, ticks
