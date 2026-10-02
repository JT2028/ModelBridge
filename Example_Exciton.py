"""
Exciton of monolayer MoS2 from the Bethe-Salpeter equation: energies at Q = 0, the exciton band dispersion, and the
wavefunction of the lowest exciton in k space and in real space. The states at Q = 0 are classified by the spin of
electron and hole (like or unlike), by their shape (s or p), and by their optical character: bright (like spin), gray
and dark (unlike spin; gray has the dipole along z). In the bands the 1s and the n = 2 states (2s, 2p) are told apart.

Run from the ModelBridge folder:    python Example_Exciton.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from Library.Bloch import hopping_blocks, bloch_hamiltonian, uniform_mesh, mesh_labels, lattice_vectors, HIGH_SYMMETRY
from Library.SpinOrbit import add_spin_orbit, SOC_STRENGTH
from Library.Exciton import screened_interaction, direct_table, converged_direct_table, load_exchange_tensor
from Library.Exciton import electron_hole_pairs, bse_hamiltonian, wavefunction_k, wavefunction_real_space
from Library.Exciton import electron_hole_distance, shortest_image, sublattice_positions
from Library.Param import N_ORBITALS


'''
# Settings
'''
mesh_size = 45                                     # k mesh, multiple of 3; 45 orders n = 2 right, 30 puts 2s below 2p
strain = {"exx": 0.0, "eyy": 0.0, "exy": 0.0}      # uniform strain of the sheet
eps_above, eps_below = 5.89, 5.89                  # dielectric constant above and below the layer (hBN); 1.0 is vacuum
with_spin = True                                   # 22 bands with spin-orbit coupling, or 11 bands without spin
converged_table = True                             # direct term taken to the limit of a fine mesh, or their recipe
tensor_file = "data/exchange_tensor_MoS2.npz"
momenta = np.linspace(-0.10, 0.10, 21)             # exciton momentum along x (1/A), for the dispersion
n_states = 20                                      # exciton states kept; 20 reach the 2s
winding_radius = 0.2                               # disc about K (1/A) in which the winding of a state is read
centre_threshold = 0.02                            # weight at K and K' above which a state is s (s > 0.09, p < 0.003)

if with_spin:
    valence_bands, conduction_bands = [12, 13], [14, 15]
else:
    valence_bands, conduction_bands = [6], [7]


'''
# Band states on the mesh, at k and at k + Q
'''
blocks = hopping_blocks(strain)
k_frac = uniform_mesh(mesh_size)
lattice, reciprocal = lattice_vectors()


def band_states(k):
    H_k = bloch_hamiltonian(blocks, k)
    if with_spin:
        H_k = add_spin_orbit(H_k, SOC_STRENGTH["Mo"], SOC_STRENGTH["S"], spin_conserving=False)
    return np.linalg.eigh(H_k)


hole_energies, hole_states = band_states(k_frac)
print(f"Band states done, gap {np.min(hole_energies[:, conduction_bands[0]] - hole_energies[:, valence_bands[-1]]):.4f} eV")


'''
# Electron-hole interaction: screened direct table, exchange tensor
'''
W = screened_interaction(eps_above, eps_below)
if converged_table:
    table = converged_direct_table(W, mesh_size)
else:
    table = direct_table(W, mesh_size)
exchange = load_exchange_tensor(tensor_file)
print("Interaction done")


'''
# Exciton at Q = 0
'''
def solve_exciton(Q_frac, with_exchange=True):
    electron_energies, electron_states = band_states(k_frac + Q_frac)
    pairs = electron_hole_pairs(hole_energies, hole_states, electron_energies, electron_states,
                                valence_bands, conduction_bands)
    H = bse_hamiltonian(pairs, table, exchange, k_frac, Q_frac, mesh_size, with_exchange=with_exchange)
    energies, amplitudes = eigh(H, subset_by_index=[0, n_states - 1])
    return energies, amplitudes, pairs


def like_spin_weight(amplitudes, pairs):
    # weight of the pairs whose electron and hole have the same spin (both up or both down), for every state
    n_up = pairs["electron"].shape[1] // 2
    up_electron = np.sum(np.abs(pairs["electron"][:, :n_up])**2, axis=1)
    up_hole = np.sum(np.abs(pairs["hole"][:, :n_up])**2, axis=1)
    like = up_electron * up_hole + (1 - up_electron) * (1 - up_hole)

    return np.abs(amplitudes.T)**2 @ like


def out_of_plane_strength(amplitudes, pairs, Q_frac):
    # |dipole along z|^2 of every state (A^2, summed over the mesh): the transition charge on the Mo atom and on the two
    # S atoms, as in exchange_long_range, times the height of each atom
    n_pairs = len(pairs["k"])
    electron = pairs["electron"].reshape(n_pairs, -1, N_ORBITALS) @ exchange["to_atoms"].T
    hole = pairs["hole"].reshape(n_pairs, -1, N_ORBITALS) @ exchange["to_atoms"].T
    atom = exchange["atom_of_orbital"]
    position = sublattice_positions(lattice)[[0, 1, 1]]
    Q = shortest_image(Q_frac, reciprocal)

    charge = np.zeros((n_pairs, 3), dtype=complex)
    for i in range(3):
        charge[:, i] = np.sum(electron[:, :, atom == i].conj() * hole[:, :, atom == i], axis=(1, 2))
        charge[:, i] *= np.exp(1j * (Q @ position[i]))

    return np.abs(amplitudes.conj().T @ (charge @ exchange["atom_height"]))**2


energies, amplitudes, pairs = solve_exciton(np.zeros(2))
energies_no_exchange, _, _ = solve_exciton(np.zeros(2), with_exchange=False)
gap = pairs["energy"].min()

print("Exciton energies at Q = 0 (eV):", np.round(energies, 4))
print("without exchange          (eV):", np.round(energies_no_exchange, 4))
print(f"Binding of the lowest exciton: {1e3 * (gap - energies_no_exchange[0]):.1f} meV")


'''
# Classify the states at Q = 0: spin sector, series, s or p, principal number, winding, optical kind
'''
if not with_spin:
    raise ValueError("The states are told apart by their spin: set with_spin = True.")


def winding_at_valley(valley, amplitudes, pairs):
    # The gauge-free envelope f(k) = A(k) <c,V|c,k> <v,k|v,V> (envelope() in 260930_example_exciton_upper_group), in a
    # disc about the valley V: its weight per state, and the weight of exp(i m theta) in it for m = -1 and +1.
    # The four band pairs of one k follow each other in pairs, hence the reshape.
    point = np.array(HIGH_SYMMETRY[valley])
    reference = band_states(point[None, :])[1][0]
    electron_overlap = np.einsum("pj,jp->p", pairs["electron"], reference[:, pairs["conduction"]].conj())
    hole_overlap = np.einsum("pj,jp->p", pairs["hole"].conj(), reference[:, pairs["valence"]])
    envelope = (amplitudes * (electron_overlap * hole_overlap)[:, None]).reshape(len(k_frac), 4, -1)

    kappa = ((k_frac - point + 0.5) % 1 - 0.5) @ reciprocal
    in_disc = np.hypot(kappa[:, 0], kappa[:, 1]) <= winding_radius
    angle = np.arctan2(kappa[:, 1], kappa[:, 0])

    weight = np.sum(np.abs(envelope[in_disc])**2, axis=(0, 1))
    share = np.zeros((2, envelope.shape[2]))
    for i, m in enumerate([-1, 1]):
        projection = np.einsum("k,kps->ps", np.exp(-1j * m * angle[in_disc]), envelope[in_disc])
        share[i] = np.sum(np.abs(projection)**2, axis=0)

    return weight, share


like = like_spin_weight(amplitudes, pairs)
sector = np.where(like > 0.5, "like", "unlike")                     # electron and hole of the same spin, or not

on_upper_valence = np.abs(amplitudes.T)**2 @ (pairs["valence"] == valence_bands[-1])
series = np.where(on_upper_valence > 0.5, "A", "B")                 # hole in the upper (A) or lower (B) valence band

valley_index = [np.flatnonzero(np.all(np.isclose(k_frac, HIGH_SYMMETRY[v]), axis=1))[0] for v in ["K", "K'"]]
centre_weight = np.sum(np.abs(amplitudes[np.isin(pairs["k"], valley_index)])**2, axis=0)
orbital = np.where(centre_weight > centre_threshold, "s", "p")      # with the hole at K or K', a p state has a node

weight_K, share_K = winding_at_valley("K", amplitudes, pairs)
weight_Kp, share_Kp = winding_at_valley("K'", amplitudes, pairs)
m_values = np.array([-1, 1])
winding = np.zeros(n_states, dtype=int)                             # m of a p state at K, 0 for an s state
for i in range(n_states):
    if orbital[i] == "p" and weight_K[i] >= weight_Kp[i]:
        winding[i] = m_values[np.argmax(share_K[:, i])]
    elif orbital[i] == "p":
        winding[i] = -m_values[np.argmax(share_Kp[:, i])]           # read at K', where the winding is reversed

# the states of one family (sector, series, shape, winding) come in pairs, the two valleys: one level = one pair
family = [(str(sector[i]), str(series[i]), str(orbital[i]), int(winding[i])) for i in range(n_states)]
levels = {}                                                         # (family, number of the level) -> its two states
principal = np.zeros(n_states, dtype=int)
for i in range(n_states):
    number = family[:i].count(family[i]) // 2
    levels.setdefault((family[i], number), []).append(i)
    principal[i] = number + (1 if orbital[i] == "s" else 2)         # 1s, 2s, ... and 2p, 3p, ...

out_of_plane_zero = out_of_plane_strength(amplitudes, pairs, np.zeros(2))
kind = np.full(n_states, "", dtype="<U6")                           # the optical kind of the s states
for (family_key, number), states in levels.items():
    if family_key[2] == "s" and family_key[0] == "like":
        kind[states] = "bright"
    elif family_key[2] == "s":
        kind[states] = "dark"
        gray_state = states[np.argmax(out_of_plane_zero[states])]   # gray has a dipole along z, dark has none
        if out_of_plane_zero[gray_state] > 1e-3 * out_of_plane_zero.max():
            kind[gray_state] = "gray"

label = []
for i in range(n_states):
    name = f"{principal[i]}s {kind[i]}" if orbital[i] == "s" else f"{principal[i]}p({winding[i]:+d}) {sector[i]}-spin"
    label.append(("B " if series[i] == "B" else "") + name)
label = np.array(label)

print("State, energy above the lowest (meV), weight at K and K', label:")
for i in range(n_states):
    print(f"{i:3d} {1e3 * (energies[i] - energies[0]):9.3f} {centre_weight[i]:10.2e}  {label[i]}")


'''
# Exciton band dispersion along x
'''
dispersion = np.zeros((len(momenta), n_states))
like_spin = np.zeros((len(momenta), n_states))         # weight of the pairs with electron and hole of the same spin
out_of_plane = np.zeros((len(momenta), n_states))      # |dipole along z|^2
for i, Q in enumerate(momenta):
    Q_frac = np.array([Q, 0.0]) @ np.linalg.inv(reciprocal)
    dispersion[i], Q_amplitudes, Q_pairs = solve_exciton(Q_frac)
    like_spin[i] = like_spin_weight(Q_amplitudes, Q_pairs)
    out_of_plane[i] = out_of_plane_strength(Q_amplitudes, Q_pairs, Q_frac)
print("Dispersion done")


'''
# Follow the states of Q = 0 along Q_x
'''
# The labels hold at Q = 0; at finite Q the states of one spin sector mix. A state is followed by its place in the
# energy order of its sector: the r-th lowest like-spin state at Q = 0 is the r-th lowest like-spin state at every Q.
zero = np.argmin(np.abs(momenta))                      # the momentum Q = 0
branch = np.zeros((n_states, len(momenta)))            # energy of every state of Q = 0 along Q_x
branch_out_of_plane = np.zeros((n_states, len(momenta)))
for i, Q in enumerate(momenta):
    for name in ["like", "unlike"]:
        at_zero = np.flatnonzero(sector == name)
        at_Q = np.flatnonzero((like_spin[i] > 0.5) == (name == "like"))
        if len(at_Q) != len(at_zero):
            raise ValueError(f"Q_x = {Q:.3f}: {len(at_Q)} {name}-spin states among {n_states}, {len(at_zero)} at Q = 0")
        branch[at_zero, i] = dispersion[i, at_Q]
        branch_out_of_plane[at_zero, i] = out_of_plane[i, at_Q]

if np.abs(branch[:, zero] - energies).max() > 1e-6:
    raise ValueError("The states followed do not start from the energies at Q = 0.")


'''
# Exciton wavefunction of the lowest state: weight in k space, electron around the hole in real space
'''
degenerate = np.flatnonzero(energies - energies[0] < 1e-3)[:2]      # the two valleys give one doublet

weight_k = np.mean([wavefunction_k(amplitudes[:, n], pairs, mesh_size) for n in degenerate], axis=0)
density = np.mean([wavefunction_real_space(amplitudes[:, n], pairs, mesh_size) for n in degenerate], axis=0)
electron_density = density.sum(axis=(0, 1))                          # electron cell from the hole's cell, all sublattices

print(f"Electron-hole distance (rms): {electron_hole_distance(density):.1f} A")
print(f"Electron on Mo: {density[0].sum():.3f}, hole on Mo: {density[:, 0].sum():.3f}")


'''
# Plot
'''
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), gridspec_kw={"width_ratios": [1.7, 1, 1]})

INK, GRY, NAVY, SKY = "0.15", "0.55", "#2E5A87", "#42B4E5"     # dark, gray, bright T, bright L
PALE_BLUE, PALE_GREY, FAINT = "#9DB8D9", "0.72", "0.85"         # 2p like-spin, 2p unlike-spin, the B series
TEXT_BLUE, TEXT_GREY = "#6A8DBB", "0.45"                        # the same two, darker for the labels
label_size = 8
ends = {}                                                       # label -> energies of its lines at the last momentum
label_colour = {}


def draw_line(energy, colour, width, order, name=None, text_colour=None):
    axes[0].plot(momenta, energy, color=colour, linewidth=width, zorder=order)
    if name is not None:
        ends.setdefault(name, []).append(energy[-1])
        label_colour[name] = text_colour if text_colour is not None else colour


for (family_key, number), (first, second) in levels.items():
    sector_name, series_name, orbital_name, m = family_key
    level = principal[first]
    lower, upper = branch[first], branch[second]

    if series_name == "B":                                      # the B series is not labelled
        draw_line(lower, FAINT, 0.8, 1)
        draw_line(upper, FAINT, 0.8, 1)
    elif orbital_name == "p" and sector_name == "like":
        name = f"{level}p like-spin (dark; m = +1 weakly bright)"
        draw_line(lower, PALE_BLUE, 0.9, 2, name, TEXT_BLUE)
        draw_line(upper, PALE_BLUE, 0.9, 2, name, TEXT_BLUE)
    elif orbital_name == "p":
        name = f"{level}p unlike-spin (dark)"
        draw_line(lower, PALE_GREY, 0.9, 2, name, TEXT_GREY)
        draw_line(upper, PALE_GREY, 0.9, 2, name, TEXT_GREY)
    elif sector_name == "like":                                 # bright: the lower state is T, the upper one L
        name = f"{level}s bright (T, L)" if level == 1 else f"{level}s bright"
        width = 1.2 if level == 1 else 1.0
        draw_line(lower, NAVY, width, 3, name)
        draw_line(upper, SKY, width, 3, name, NAVY)
    else:                                                       # the state with the larger dipole along z is gray
        second_is_gray = branch_out_of_plane[second] > branch_out_of_plane[first]
        gray_line = np.where(second_is_gray, upper, lower)
        dark_line = np.where(second_is_gray, lower, upper)
        draw_line(dark_line, INK, 2.6 if level == 1 else 1.0, 3, f"{level}s dark")
        draw_line(gray_line, GRY, 1.0, 3, f"{level}s gray")

axes[0].set_xlim(momenta[0] - 0.005, momenta[-1] + 0.13)
axes[0].set_xticks(np.linspace(momenta[0], momenta[-1], 5))
axes[0].set_xlabel(r"Exciton momentum $Q_x$ (1/$\mathrm{\AA}$)")
axes[0].set_ylabel("Exciton energy (eV)")
axes[0].set_title("Exciton bands")
axes[0].grid(True, linestyle="--", alpha=0.5)

k_cart = k_frac @ reciprocal
axes[1].scatter(k_cart[:, 0], k_cart[:, 1], c=weight_k.ravel(), s=12, cmap="viridis")
axes[1].set_aspect("equal")
axes[1].set_xlabel(r"$k_x$ (1/$\mathrm{\AA}$)")
axes[1].set_ylabel(r"$k_y$ (1/$\mathrm{\AA}$)")
axes[1].set_title(r"$|A(k)|^2$ of the lowest exciton")

cells = (mesh_labels(mesh_size) + mesh_size // 2) % mesh_size - mesh_size // 2     # counted from the hole
position = cells @ lattice
axes[2].scatter(position[:, 0], position[:, 1], c=electron_density.ravel(), s=25, cmap="viridis")
axes[2].set_aspect("equal")
axes[2].set_xlim(-30, 30)
axes[2].set_ylim(-30, 30)
axes[2].set_xlabel(r"x ($\mathrm{\AA}$)")
axes[2].set_ylabel(r"y ($\mathrm{\AA}$)")
axes[2].set_title("Electron around the hole")

plt.tight_layout()

# the labels stand right of the lines, one per group, pushed apart where two lie closer than the height of a label
bottom, top = axes[0].get_ylim()
axes_height = axes[0].get_position().height * fig.get_figheight() * 72          # points
label_gap = 1.4 * label_size / axes_height * (top - bottom)                   # eV
label_x = momenta[-1] + 0.008

names = sorted(ends, key=lambda name: np.mean(ends[name]))
line_height = np.array([np.mean(ends[name]) for name in names])
text_height = line_height.copy()
for _ in range(200):
    for j in range(len(names) - 1):
        too_close = label_gap - (text_height[j + 1] - text_height[j])
        if too_close > 0:
            text_height[j] -= too_close / 2
            text_height[j + 1] += too_close / 2

for name, y_line, y_text in zip(names, line_height, text_height):
    axes[0].text(label_x, y_text, name, color=label_colour[name], fontsize=label_size, va="center",
                 bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    axes[0].plot([momenta[-1] + 0.001, label_x - 0.002], [y_line, y_text], color=label_colour[name], linewidth=0.5)

plt.savefig("exciton.png", dpi=300)
plt.close()

np.savez("exciton.npz", momenta=momenta, dispersion=dispersion, energies=energies,
         energies_no_exchange=energies_no_exchange, k_cart=k_cart, weight_k=weight_k, position=position,
         electron_density=electron_density, like_spin=like_spin, out_of_plane=out_of_plane, branch=branch,
         branch_out_of_plane=branch_out_of_plane, sector=sector, series=series, orbital=orbital, principal=principal,
         winding=winding, kind=kind, label=label, centre_weight=centre_weight)
print("Exciton done")
