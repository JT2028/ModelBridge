"""
Exciton of the monolayer: Bethe-Salpeter equation in momentum space (Tamm-Dancoff), with the electron-hole interaction
of Cheng's group (WannierBSE: Peng 2019, Shih 2025).

A pair is an electron in a conduction band at k + Q and a hole in a valence band at k, with Q the exciton's momentum:

    H = (E_electron - E_hole) - direct + spin_factor * exchange        spin_factor = 2 without spin, 1 with spin

direct      screened interaction of the dielectric stack between charges on the Mo and S sites
exchange    bare interaction between orbital pair densities: their tensor (short range) and the G = 0 term (long range)

Energies in eV, lengths in Angstrom, momenta fractional as in Bloch.py. States have 11 orbitals, or 22 with spin
(spin-major, as SpinOrbit.py returns them).

With spin, the exchange acts on the pair density summed over spin, through the orbital part of their tensor (the mean
of its spin blocks). The spin-dependent part of their tensor, 0.7 % of its entries, is left out.
"""

import numpy as np
from .Bloch import LATTICE_CONSTANT, lattice_vectors, mesh_labels
from .Param import N_ORBITALS

COULOMB = 14.399645           # e^2 / (4 pi eps0), eV A
HBAR2_OVER_2M = 3.80998212    # hbar^2 / (2 m0), eV A^2

# dielectric model of the MoS2 layer (Shih 2025, Eq. 9; values of WannierBSE's MoS2 examples)
SLAB_THICKNESS = 6.26         # A
BULK_DIELECTRIC = 13.24
PLASMON_ENERGY = 22.5         # eV
THOMAS_FERMI_Q = 2.308694     # 1/A, from the plasmon energy
ALPHA = 1.55

SUBLATTICE = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])    # of each orbital: 0 = Mo, 1 = S
S_POSITION = np.array([-2.0 / 3.0, -1.0 / 3.0])             # of the S pair from Mo, in units of (a1, a2)


def sublattice_positions(lattice):
    return np.array([[0.0, 0.0], S_POSITION @ lattice])


def image_offsets(reach):
    offsets = np.arange(-reach, reach + 1)

    return np.array([[i, j] for i in offsets for j in offsets])


'''
================================================
Below is for the screened interaction
================================================
'''
def dielectric_bulk(q):
    inverse = 1.0 / (BULK_DIELECTRIC - 1.0)
    inverse += ALPHA * q**2 / THOMAS_FERMI_Q**2
    inverse += (HBAR2_OVER_2M * q**2 / PLASMON_ENERGY)**2

    return 1.0 + 1.0 / inverse


def slab_coulomb(q):
    x = q * SLAB_THICKNESS

    return 4 * np.pi / (SLAB_THICKNESS * q**2) * (1.0 - (1.0 - np.exp(-x)) / x)


def dielectric_stack(q, eps_above, eps_below, n_heights=48):
    half = SLAB_THICKNESS / 2
    nodes, weights = np.polynomial.legendre.leggauss(n_heights)
    source = nodes * half
    weights = weights * half

    eps_layer = dielectric_bulk(q)
    up = np.exp(-q * (half - source))
    down = np.exp(-q * (half + source))

    # potential of a charge at the height `source`, for every height at once:
    #     inside the layer  A e^{qz} + B e^{-qz} + source term;    above  C e^{-qz};    below  D e^{qz}
    # unknowns (A, B, C, D). Rows 0, 1: potential and displacement continuous at the top face, where D does not enter;
    # rows 2, 3: the same at the bottom face, where C does not enter
    matrix = np.zeros((n_heights, 4, 4))
    rhs = np.zeros((n_heights, 4))

    matrix[:, 0, :3] = [np.exp(q * half), np.exp(-q * half), -np.exp(-q * half)]
    matrix[:, 1, :3] = [eps_layer * np.exp(q * half), -eps_layer * np.exp(-q * half), eps_above * np.exp(-q * half)]
    matrix[:, 2, [0, 1, 3]] = [np.exp(-q * half), np.exp(q * half), -np.exp(-q * half)]
    matrix[:, 3, [0, 1, 3]] = [eps_layer * np.exp(-q * half), -eps_layer * np.exp(q * half), -eps_below * np.exp(-q * half)]

    rhs[:, 0] = -up / (2 * q * eps_layer)
    rhs[:, 1] = up / (2 * q)
    rhs[:, 2] = -down / (2 * q * eps_layer)
    rhs[:, 3] = -down / (2 * q)

    solution = np.linalg.solve(matrix, rhs[:, :, None])
    A = solution[:, 0, 0]
    B = solution[:, 1, 0]

    homogeneous = (A + B) * 2 * np.sinh(q * half) / q
    particular = (2 - up - down) / (2 * q**2 * eps_layer)

    screened = np.sum(weights * (homogeneous + particular))
    bare = (q * SLAB_THICKNESS - 1 + np.exp(-q * SLAB_THICKNESS)) / q**3

    return bare / screened


def screened_interaction(eps_above, eps_below, q_max=6.0, n_q=1201):
    q_grid = np.linspace(0.0, q_max, n_q)

    eps_grid = np.zeros(n_q)
    eps_grid[0] = 0.5 * (eps_above + eps_below)
    eps_grid[1:] = [dielectric_stack(q, eps_above, eps_below) for q in q_grid[1:]]

    def W(q):
        return COULOMB * slab_coulomb(q) / np.interp(q, q_grid, eps_grid)

    return W


'''
================================================
Below is for the direct interaction between sublattices
================================================
'''
def cell_average(W, reciprocal, mesh_size, n_radial=400, n_angular=240):
    # $$\bar W = \frac{6}{A_{\rm hexagon}} \int_0^{\pi/3} d\theta \int_0^{r_{\max}(\theta)} W(r)\; r\, dr, \qquad r_{\max}(\theta) = \frac{r_{\rm in}}{\cos(\theta - \pi/6)}$$
    # average over small q, direct interaction to avoid singularities at q = 0
    
    inradius = 0.5 * np.linalg.norm(reciprocal[0]) / mesh_size
    cell_area = abs(np.linalg.det(reciprocal)) / mesh_size**2

    r_nodes, r_weights = np.polynomial.legendre.leggauss(n_radial)
    t_nodes, t_weights = np.polynomial.legendre.leggauss(n_angular)
    angles = (t_nodes + 1) * np.pi / 6

    total = 0.0
    for angle, weight in zip(angles, t_weights * np.pi / 6):
        r_max = inradius / np.cos(angle - np.pi / 6)
        r = (r_nodes + 1) * r_max / 2

        total += weight * (r_max / 2) * np.sum(r_weights * r * W(r))

    return 6 * total / cell_area


def direct_table(W, mesh_size, a=LATTICE_CONSTANT):
    lattice, reciprocal = lattice_vectors(a)
    unit_cell_area = abs(np.linalg.det(lattice))
    
    # q = (m1·b1 + m2·b2)/N in 1/Å
    q = mesh_labels(mesh_size) @ reciprocal / mesh_size

    # shift reciprocal vectors G = i·b1 + j·b2, with i and j from −2 to 2
    shifts = image_offsets(2) @ reciprocal
    images = q[:, None, :] - shifts[None, :, :]
    lengths = np.linalg.norm(images, axis=-1)
    shortest = lengths.min(axis=1)
    tied = lengths < shortest[:, None] + 1e-9
    # get the shortest reciprocal vector for each q point in hexagonal Brillouin zone

    strength = np.zeros(len(q))
    strength[shortest > 1e-12] = W(shortest[shortest > 1e-12])
    strength[shortest <= 1e-12] = cell_average(W, reciprocal, mesh_size)

    position = sublattice_positions(lattice)

    table = np.zeros((2, 2, len(q)), dtype=complex)

    for s in range(2):
        for t in range(2):
            phase = np.exp(-1j * (images @ (position[s] - position[t])))
            phase = (phase * tied).sum(axis=1) / tied.sum(axis=1)

            table[s, t] = strength * phase / unit_cell_area

    return table


def converged_direct_table(W, mesh_size, fine_mesh_size=210, a=LATTICE_CONSTANT):
    if fine_mesh_size < 2 * mesh_size:
        raise ValueError(f"fine_mesh_size = {fine_mesh_size} is below twice the mesh size {mesh_size}.")

    lattice, _ = lattice_vectors(a)

    between_sites = {}
    for size in (fine_mesh_size, 2 * fine_mesh_size):
        fine = direct_table(W, size, a).reshape(2, 2, size, size)
        between_sites[size] = np.fft.fft2(fine, axes=(2, 3)).real / size**2

    labels = mesh_labels(mesh_size)
    images = labels[:, None, :] + mesh_size * image_offsets(1)[None, :, :]

    position = sublattice_positions(lattice)
    to_momentum = np.exp(2j * np.pi * (labels @ labels.T) / mesh_size)

    table = np.zeros((2, 2, len(labels)), dtype=complex)

    for s in range(2):
        for t in range(2):
            distance = np.linalg.norm(images @ lattice + position[s] - position[t], axis=-1)
            nearest = images[np.arange(len(labels)), np.argmin(np.round(distance, 9), axis=1)]

            # the error of the recipe falls as 1 / mesh size: two meshes, extrapolated
            values = []
            for size in (fine_mesh_size, 2 * fine_mesh_size):
                values.append(between_sites[size][s, t][nearest[:, 0] % size, nearest[:, 1] % size])

            table[s, t] = to_momentum @ (2 * values[1] - values[0])

    return table


'''
================================================
Below is for the electron-hole pairs and the three terms of H
================================================
'''
def electron_hole_pairs(hole_energies, hole_states, electron_energies, electron_states, valence_bands, conduction_bands):
    n_k = len(hole_energies)
    # lists every electron–hole pair 
    k = np.repeat(np.arange(n_k), len(conduction_bands) * len(valence_bands))
    conduction = np.tile(np.repeat(conduction_bands, len(valence_bands)), n_k)
    valence = np.tile(valence_bands, n_k * len(conduction_bands))

    pairs = {
        "k": k,
        "conduction": conduction,
        "valence": valence,
        "electron": electron_states[k, :, conduction],
        "hole": hole_states[k, :, valence],
        "energy": electron_energies[k, conduction] - hole_energies[k, valence],
    }

    return pairs


def direct_term(pairs, table, mesh_size):
    n_orbitals = pairs["electron"].shape[1]
    sublattice = np.tile(SUBLATTICE, n_orbitals // N_ORBITALS)

    # momentum transfer k' - k between the pair of the row (k) and the pair of the column (k'), as a mesh index
    row, col = np.divmod(pairs["k"].astype(np.int32), mesh_size)
    q_row = (row[None, :] - row[:, None]) % mesh_size
    q_col = (col[None, :] - col[:, None]) % mesh_size
    q = q_row * mesh_size + q_col

    D = np.zeros(q.shape, dtype=complex)

    for s in range(2):
        electron = pairs["electron"][:, sublattice == s]
        electron_overlap = electron.conj() @ electron.T

        for t in range(2):
            hole = pairs["hole"][:, sublattice == t]
            hole_overlap = hole @ hole.conj().T

            D += electron_overlap * table[s, t][q] * hole_overlap

    return D / mesh_size**2


def shortest_image(k_frac, reciprocal):
    # for a momentum exactly on the zone boundary the shortest image is not unique, and one of them is returned
    images = (np.asarray(k_frac) - np.round(k_frac) + image_offsets(1)) @ reciprocal

    return images[np.argmin(np.linalg.norm(images, axis=1))]


def load_exchange_tensor(filename, symmetric=False):
    data = np.load(filename)

    exchange = {key: data[key] for key in ("cells", "cell_area", "to_atoms", "atom_of_orbital", "atom_height")}
    exchange["tensor"] = data["tensor_symmetric"] if symmetric else data["tensor"]

    return exchange


def exchange_short_range(pairs, exchange, k_frac, Q_frac):
    n_pairs = len(pairs["k"])
    electron = pairs["electron"].reshape(n_pairs, -1, N_ORBITALS)
    hole = pairs["hole"].reshape(n_pairs, -1, N_ORBITALS)

    # the hole's orbital stays in the home cell; the electron's orbital sits in the cell R of the tensor
    electron_k = k_frac[pairs["k"]] + np.asarray(Q_frac)
    phase = np.exp(-2j * np.pi * (electron_k @ exchange["cells"].T))

    # pair density between two orbitals, summed over spin
    density = np.einsum("psc,psd->pcd", hole, electron.conj())
    density = (phase[:, :, None, None] * density[:, None, :, :]).reshape(n_pairs, -1)

    return (density @ exchange["tensor"]) @ density.conj().T / exchange["cell_area"]


def exchange_long_range(pairs, exchange, Q_frac, a=LATTICE_CONSTANT):
    lattice, reciprocal = lattice_vectors(a)
    Q = shortest_image(Q_frac, reciprocal)
    Q_length = np.linalg.norm(Q)

    if Q_length < 1e-12:
        return 0.0

    n_pairs = len(pairs["k"])
    electron = pairs["electron"].reshape(n_pairs, -1, N_ORBITALS) @ exchange["to_atoms"].T
    hole = pairs["hole"].reshape(n_pairs, -1, N_ORBITALS) @ exchange["to_atoms"].T

    # point charges on the three atoms (Mo, upper S, lower S) at their heights
    atom = exchange["atom_of_orbital"]
    height = exchange["atom_height"]
    position = sublattice_positions(lattice)[[0, 1, 1]]

    charge = np.zeros((n_pairs, 3), dtype=complex)
    for i in range(3):
        charge[:, i] = np.sum(electron[:, :, atom == i].conj() * hole[:, :, atom == i], axis=(1, 2))
        charge[:, i] *= np.exp(1j * (Q @ position[i]))

    coulomb = 2 * np.pi * COULOMB / Q_length * np.exp(-Q_length * np.abs(height[:, None] - height[None, :]))

    return (charge @ coulomb) @ charge.conj().T / abs(np.linalg.det(lattice))


def exchange_term(pairs, exchange, k_frac, Q_frac, mesh_size, a=LATTICE_CONSTANT, long_range=True):
    X = exchange_short_range(pairs, exchange, k_frac, Q_frac)

    if long_range:
        X += exchange_long_range(pairs, exchange, Q_frac, a)

    return X / mesh_size**2


def bse_hamiltonian(pairs, table, exchange, k_frac, Q_frac, mesh_size, a=LATTICE_CONSTANT, with_exchange=True):
    has_spin = pairs["electron"].shape[1] == 2 * N_ORBITALS
    spin_factor = 1.0 if has_spin else 2.0

    H = np.diag(pairs["energy"]).astype(complex)
    H -= direct_term(pairs, table, mesh_size)

    if with_exchange:
        H += spin_factor * exchange_term(pairs, exchange, k_frac, Q_frac, mesh_size, a)

    return H


'''
================================================
Below is for the exciton wavefunction
================================================
'''
def wavefunction_k(amplitude, pairs, mesh_size):
    weight = np.zeros(mesh_size**2)
    np.add.at(weight, pairs["k"], np.abs(amplitude)**2)

    return weight.reshape(mesh_size, mesh_size)


def wavefunction_real_space(amplitude, pairs, mesh_size):
    n_orbitals = pairs["electron"].shape[1]
    sublattice = np.tile(SUBLATTICE, n_orbitals // N_ORBITALS)

    products = amplitude[:, None, None] * pairs["electron"][:, :, None] * pairs["hole"].conj()[:, None, :]

    pair_amplitude = np.zeros((mesh_size**2, n_orbitals, n_orbitals), dtype=complex)
    np.add.at(pair_amplitude, pairs["k"], products)

    pair_amplitude = pair_amplitude.reshape(mesh_size, mesh_size, n_orbitals, n_orbitals)
    cell_amplitude = np.fft.ifft2(pair_amplitude, axes=(0, 1))
    probability = mesh_size**2 * np.abs(cell_amplitude)**2

    density = np.zeros((2, 2, mesh_size, mesh_size))
    for s in range(2):
        for t in range(2):
            density[s, t] = probability[:, :, sublattice == s][:, :, :, sublattice == t].sum(axis=(2, 3))

    return density


def electron_hole_distance(density, a=LATTICE_CONSTANT):
    lattice, _ = lattice_vectors(a)
    mesh_size = density.shape[-1]

    cells = mesh_labels(mesh_size)[:, None, :] + mesh_size * image_offsets(1)[None, :, :]
    position = sublattice_positions(lattice)

    mean_square = 0.0
    for s in range(2):
        for t in range(2):
            images = cells @ lattice + position[s] - position[t]
            nearest = np.linalg.norm(images, axis=-1).min(axis=1)

            mean_square += np.sum(density[s, t].ravel() * nearest**2)

    return np.sqrt(mean_square)
