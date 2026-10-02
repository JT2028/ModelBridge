"""
Band unfolding by FFT: the band structure inside the Hamiltonian of a periodic real-space supercell, laid out as
build_full_H lays it out. It takes the eigenstates of build_full_H, of build_full_H_gw, and of either one after
add_spin_orbit_real_space.

A periodic N x N supercell allows N * N momenta, k_frac = (m1, m2) / N. A Bloch state is a plane wave over the cells,
so the FFT of an eigenvector over the cell labels puts its weight on the mesh point it carries:

    weights[m1, m2, state] = (1 / n_cells) sum_spin,orbital |sum_cells psi exp(-2 pi i (m1, m2) . cell / N)|^2

Summed over the mesh it is 1 for every state. Summed over the states it is the number of orbitals per cell at every
mesh point, 11 without spin and 22 with spin. weight_sum_error returns the largest deviation from these two sums; it
needs every state of the supercell. On a flat supercell every state sits on a band of the Bloch Hamiltonian at the
momentum it carries. On a strained one its weight spreads over the mesh and its energy leaves the flat bands;
band_offset measures by how much.

States are the columns of eigh(H), index = spin * n_spinless + cell * 11 + orbital. eigh mixes degenerate states
freely, and k and -k are always degenerate, so band_dots adds the weights of each degenerate group.
K is on the mesh when 3 divides N, M when N is even.
"""

import numpy as np
from .Bloch import HIGH_SYMMETRY, LATTICE_CONSTANT, lattice_vectors
from .Param import N_ORBITALS

ON_PATH = 1e-9    # a mesh point counts as on the path within this distance, in fractional momentum


def cell_labels(cells):
    cells = cells.sort_values("cell")

    label_x = cells["cell_x"].to_numpy(dtype=int)
    label_y = cells["cell_y"].to_numpy(dtype=int)

    label_x = label_x - label_x.min()
    label_y = label_y - label_y.min()

    supercell_side = int(max(label_x.max(), label_y.max())) + 1

    if len(set(zip(label_x, label_y))) != supercell_side**2 or len(cells) != supercell_side**2:
        raise ValueError(f"Unfolding needs a full square block of cells; found {len(cells)} cells, side {supercell_side}.")

    return label_x, label_y, supercell_side


'''
================================================
Below is for the weight of every state on the momentum mesh
================================================
'''
def unfold_weights(states, cells, chunk_size=512):
    label_x, label_y, supercell_side = cell_labels(cells)

    n_cells = len(label_x)
    n_states = states.shape[1]
    n_spin = states.shape[0] // (n_cells * N_ORBITALS)

    if n_spin not in (1, 2) or states.shape[0] != n_spin * n_cells * N_ORBITALS:
        raise ValueError(f"states has {states.shape[0]} rows; expected {N_ORBITALS} or {2 * N_ORBITALS} per cell.")

    weights = np.empty((supercell_side, supercell_side, n_states))

    for start in range(0, n_states, chunk_size):
        stop = min(start + chunk_size, n_states)

        chunk = np.asarray(states[:, start:stop]).reshape(n_spin, n_cells, N_ORBITALS, stop - start)

        on_lattice = np.zeros((supercell_side, supercell_side, n_spin, N_ORBITALS, stop - start), dtype=complex)
        on_lattice[label_x, label_y] = chunk.transpose(1, 0, 2, 3)

        amplitude = np.fft.fft2(on_lattice, axes=(0, 1))

        weights[:, :, start:stop] = np.sum(np.abs(amplitude)**2, axis=(2, 3)) / n_cells

    return weights


def weight_sum_error(weights):
    n_mesh = weights.shape[0] * weights.shape[1]
    n_orbitals_per_cell = weights.shape[2] / n_mesh

    over_mesh = np.abs(weights.sum(axis=(0, 1)) - 1.0).max()
    over_states = np.abs(weights.sum(axis=2) - n_orbitals_per_cell).max()

    return max(over_mesh, over_states)


def band_offset(weights, energies, bands_on_mesh):
    n_mesh = weights.shape[0] * weights.shape[1]

    weights = weights.reshape(n_mesh, -1)
    offset = np.zeros(len(energies))

    for point in range(n_mesh):
        to_nearest_band = np.abs(bands_on_mesh[point][:, None] - energies[None, :]).min(axis=0)

        offset += weights[point] * to_nearest_band

    return offset


'''
================================================
Below is for the unfolded bands along a path
================================================
'''
def path_on_mesh(supercell_side, point_names, a=LATTICE_CONSTANT):
    _, reciprocal = lattice_vectors(a)

    corners = np.array([HIGH_SYMMETRY[name] for name in point_names])
    n_segments = len(corners) - 1

    n_labels = 2 * supercell_side + 1
    labels = np.indices((n_labels, n_labels)).reshape(2, -1).T - supercell_side

    mesh_index = []
    distance = []
    segment_start = 0.0

    for segment in range(n_segments):
        direction = corners[segment + 1] - corners[segment]
        from_corner = labels / supercell_side - corners[segment]

        position = from_corner @ direction / (direction @ direction)
        off_line = from_corner[:, 0] * direction[1] - from_corner[:, 1] * direction[0]

        # a corner shared by two segments is kept once, as the start of the second
        segment_end = 1.0 + ON_PATH if segment == n_segments - 1 else 1.0 - ON_PATH
        on_segment = (np.abs(off_line) < ON_PATH) & (position > -ON_PATH) & (position < segment_end)

        along = np.where(on_segment)[0][np.argsort(position[on_segment])]
        length = np.linalg.norm(direction @ reciprocal)

        mesh_index.extend(labels[along] % supercell_side)
        distance.extend(segment_start + position[along] * length)

        segment_start += length

    return np.array(mesh_index), np.array(distance)


def band_dots(weights_on_path, energies, distance, weight_cut=1e-3, degeneracy=1e-6):
    group = np.concatenate([[0], np.cumsum(np.diff(energies) > degeneracy)])

    group_energy = np.bincount(group, weights=energies) / np.bincount(group)
    group_weight = np.array([np.bincount(group, weights=row) for row in weights_on_path])

    point, level = np.nonzero(group_weight > weight_cut)

    return distance[point], group_energy[level], group_weight[point, level]


def unfold_bands(energies, states, cells, point_names, a=LATTICE_CONSTANT):
    weights = unfold_weights(states, cells)

    mesh_index, distance = path_on_mesh(weights.shape[0], point_names, a)
    weights_on_path = weights[mesh_index[:, 0], mesh_index[:, 1]]

    dot_distance, dot_energy, dot_weight = band_dots(weights_on_path, energies, distance)

    return {
        "weights": weights,
        "mesh_index": mesh_index,
        "distance": distance,
        "dot_distance": dot_distance,
        "dot_energy": dot_energy,
        "dot_weight": dot_weight,
    }
