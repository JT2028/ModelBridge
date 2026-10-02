"""
GW correction of the MoS2 tight-binding Hamiltonian (Fang 2015, Table IX): every on-site energy is shifted and every
hopping is scaled, t_GW = z t_DFT.

scale_strain=True scales the strained hopping as a whole; False scales only its zero-strain part (the paper gives the
factors for the unstrained sheet only).
"""

import numpy as np
from .Bloch import NO_STRAIN
from .Hamiltonian import build_H, wrap
from .Neighbor import orb_i, C3_half_neighbor_list
from .Param import GROUPS, N_ORBITALS, VALID

# Fang 2015 Table IX, MoS2
GW_ONSITE_SHIFT = {"metal": 0.3624, "chalcogen": -0.2512}    # eV

GW_HOPPING_SCALE = {
    "1st M-M": 1.4209,
    "1st X-X": 1.1738,
    "1st X-M": 1.0773,
    "2nd X-M": 1.1871,
}

SPECIES = {"A": "metal", "B": "chalcogen", "C": "metal", "D": "chalcogen"}


def hopping_scale(order, alpha, beta):
    if (alpha, beta) not in VALID[order]:
        raise ValueError(f"Invalid block pair for order {order}: {(alpha, beta)}")

    if order == 1:
        return GW_HOPPING_SCALE["1st X-M"]

    if order == 2 and SPECIES[alpha] == "metal":
        return GW_HOPPING_SCALE["1st M-M"]

    if order == 2:
        return GW_HOPPING_SCALE["1st X-X"]

    if order == 3:
        return GW_HOPPING_SCALE["2nd X-M"]

    raise ValueError(f"Unknown neighbor order: {order}")


def build_H_gw(instr, strain, scale_strain=True):
    order = instr["order"]
    alpha = instr["alpha"]
    beta = instr["beta"]

    H_dft = build_H(instr, strain)

    if order == 0:
        return H_dft + GW_ONSITE_SHIFT[SPECIES[alpha]] * np.eye(H_dft.shape[0])

    scale = hopping_scale(order, alpha, beta)

    if scale_strain:
        return scale * H_dft

    return H_dft + (scale - 1.0) * build_H(instr, NO_STRAIN)


def average_strain(strain_1, strain_2):
    return {key: 0.5 * (strain_1[key] + strain_2[key]) for key in ("exx", "eyy", "exy")}


def no_wrap(tar):
    return tar


def assemble_H_gw(cells, strain_df, scale_strain, wrap_cell):
    strain_df = strain_df.set_index("id")

    mo_id_lookup = {(row.cell_x, row.cell_y): row.mo_id for row in cells.itertuples(index=False)}
    cell_lookup = {(row.cell_x, row.cell_y): row.cell for row in cells.itertuples(index=False)}

    n_cells = len(cells)
    H = np.zeros((n_cells * N_ORBITALS, n_cells * N_ORBITALS), dtype=complex)

    for cell in cells.itertuples(index=False):
        nx = cell.cell_x
        ny = cell.cell_y

        src_strain = strain_df.loc[cell.mo_id, ["exx", "eyy", "exy"]].to_dict()

        for alpha in GROUPS:
            instr = {"order": 0, "alpha": alpha, "beta": alpha, "c3": 0}

            idx = orb_i(cell_lookup, nx, ny, alpha)
            H[np.ix_(idx, idx)] += build_H_gw(instr, src_strain, scale_strain)

        for order, Hn in zip([1, 2, 3], C3_half_neighbor_list(nx, ny)):
            for tar, c3 in Hn:
                mx, my = wrap_cell(tar)

                if (mx, my) not in mo_id_lookup:
                    continue

                tar_strain = strain_df.loc[mo_id_lookup[(mx, my)], ["exx", "eyy", "exy"]].to_dict()
                bond_strain = average_strain(src_strain, tar_strain)

                for alpha, beta in sorted(VALID[order]):
                    instr = {"order": order, "alpha": alpha, "beta": beta, "c3": c3}

                    H_block = build_H_gw(instr, bond_strain, scale_strain)

                    row_idx = orb_i(cell_lookup, nx, ny, alpha)
                    col_idx = orb_i(cell_lookup, mx, my, beta)

                    H[np.ix_(row_idx, col_idx)] += H_block
                    H[np.ix_(col_idx, row_idx)] += H_block.conj().T

    return H


'''
================================================
Below is for periodic neighbors
================================================
'''
def build_full_H_gw(cells, strain_df, scale_strain=True):
    return assemble_H_gw(cells, strain_df, scale_strain, wrap(cells))


'''
================================================
Below is for non-periodic neighbors
================================================
'''
def build_full_H_gw_open(cells, strain_df, scale_strain=True):
    return assemble_H_gw(cells, strain_df, scale_strain, no_wrap)
