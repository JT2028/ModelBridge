"""Unfold_MoS2_Au_calculation.py — band structure inside the tight-binding window of one 260910 MoS2/Au(111) MD run.
Same structure as LDOS_MoS2_Au_calculation.py. One copy lives in each MD run folder and
analyses the folder it sits in:

    cd /media/yehlab/C/JenTe/260910_MoS2_Au_MD/MoS2_Au_flat && python3 Unfold_MoS2_Au_calculation.py

Steps (each is one ModelBridge call): frames -> strain (compute_strain_tensor_from_frames) ->
Mo cells (extract_mo_cells, assign_cell_indices) -> N x N window (select_matrix_cells) ->
H (build_full_H, periodic, kept real; with gw: build_full_H_gw; with spin_orbit: add_spin_orbit_real_space) ->
full eigh -> FFT of every eigenvector over the cells (unfold_bands) -> bands of the flat sheet straight from
k space (bloch_hamiltonian) -> files in OUT_DIR (unfold.npz, fig_unfolded_bands.png, log.txt).

One figure, two band structures:
    dots   the states of the window at the mesh points on Γ-M-K-Γ (dot area = weight)
    lines  the flat sheet, H(k) diagonalised at each k, with the same gw and spin_orbit

`Library/` and `find_frames.py` come from the ModelBridge clone in MODELBRIDGE (below);
Library/Bloch.py, SpinOrbit.py, GW.py and Unfold.py must be in it.
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# let the script find the source code
MODELBRIDGE = os.environ.get("MODELBRIDGE", "/media/yehlab/C/JenTe/ModelBridge")
sys.path.insert(0, MODELBRIDGE)

from Library.Neighbor import read_lammps_steps, extract_mo_cells, assign_cell_indices, select_matrix_cells  # noqa: E402
from Library.Hamiltonian import build_full_H                      # noqa: E402
from Library.Strain import compute_strain_tensor_from_frames      # noqa: E402
from Library.Bloch import hopping_blocks, bloch_hamiltonian, uniform_mesh, k_path   # noqa: E402
from Library.SpinOrbit import add_spin_orbit, add_spin_orbit_real_space, SOC_STRENGTH  # noqa: E402
from Library.GW import build_H_gw, build_full_H_gw                # noqa: E402
from Library.Unfold import unfold_bands, band_offset, weight_sum_error   # noqa: E402
from find_frames import pick_frames                               # noqa: E402

t0 = t_last = time.time()

# the MD folder to analyse: the one this copy of the script sits in
RUN_DIR = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(RUN_DIR, "positions.dat")

Mo_types = [2, 5, 8, 11]                 # atom types by mass in initial_positions.txt: Mo 95.94
nei_radius = 8.0                         # strain-fit neighbour cutoff (A)
a = 3.117                                # Mo-Mo lattice constant for cell labelling (A); also the scale of the k axis
supercell_side = 21                      # window around cell (0, 0) = the pillar axis; K is on the k mesh when 3 divides it
spin_orbit = False                       # True: lambda L.S on every atom (Fang 2015 Table VIII), H is 22 per cell, complex
gw = False                               # True: GW on-site shifts and hopping factors (Fang 2015 Table IX)
path = ["G", "M", "K", "G"]              # corners of the k path, names of Library.Bloch.HIGH_SYMMETRY
path_labels = ["Γ", "M", "K", "Γ"]       # the same corners as written on the figure
E_window = (-8.0, -2.0)                  # energy range of the figure (eV)

# find the reference and deformed frames from "lj/cut" in log.lammps
ref_step, def_step = pick_frames(RUN_DIR)
print(f"Frames picked from log.lammps / positions.dat: reference {ref_step}, deformed {def_step}")

# create a subfolder for the outputs, named after the window and the model
model = ("_gw" if gw else "") + ("_soc" if spin_orbit else "")
OUT_DIR = os.path.join(RUN_DIR, f"unfold_{supercell_side}x{supercell_side}{model}")
NAME = os.path.basename(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# helper name function for outputs to be saved in OUT_DIR
P = lambda s: os.path.join(OUT_DIR, f"{NAME}_{s}")                # noqa: E731

# log.txt record some prarmeters and time
n_orb = 22 if spin_orbit else 11                                  # states per cell
n_val = 14 if spin_orbit else 7                                   # of which filled (valence bands)
LOG = {"supercell": {"side_cells": supercell_side, "n_cells": supercell_side ** 2,
                     "H_dim": n_orb * supercell_side ** 2, "a_A": a},
       "model": {"spin_orbit": spin_orbit, "gw": gw},
       "frames": {"ref_step": ref_step, "def_step": def_step, "nei_radius_A": nei_radius},
       "timing_s": {}}


# log function
def lap(section):
    global t_last
    now = time.time()
    LOG["timing_s"].pop("total", None)
    LOG["timing_s"][section] = round(now - t_last, 1)
    LOG["timing_s"]["total"] = round(now - t0, 1)
    t_last = now
    with open(P("log.txt"), "w") as f:
        json.dump(LOG, f, indent=2)


lap("initialization")

'''
# Building strained Hamiltonian
'''
ref_df, def_df = read_lammps_steps(filename, [ref_step, def_step])
print("Reference and Deformed Frames loaded")
lap("load_frames")

strain_df = compute_strain_tensor_from_frames(ref_df, def_df, atom_types=Mo_types, cutoff_radius=nei_radius)
print(f"Strain tensor constructed: {len(strain_df)} Mo atoms, NaN rows {int(strain_df['exx'].isna().sum())}")
lap("compute_strain")

mo_df = extract_mo_cells(ref_df, mo_types=tuple(Mo_types))
cells = assign_cell_indices(mo_df, a=a, theta_deg=0, search_radius=1.5)
selected = select_matrix_cells(cells, chosen_cell_x=0, chosen_cell_y=0, supercell_side=supercell_side)
n_cells = len(selected)
lap("cells_construction")

H = build_full_H_gw(selected, strain_df) if gw else build_full_H(selected, strain_df)
if spin_orbit:
    H = add_spin_orbit_real_space(H, SOC_STRENGTH["Mo"], SOC_STRENGTH["S"])     # complex, 22 per cell
else:
    # without spin H is real symmetric. Keep it real: eigh then uses the real solver
    # (LDOS_MoS2_Au_calculation.py: 14 s instead of 127 s for the 21x21 window on sim).
    if np.abs(H.imag).max() != 0:
        raise ValueError("H has a nonzero imaginary part; keep it complex")
    H = np.ascontiguousarray(H.real)
print("H constructed")
print("H shape:", H.shape)
print("H dtype:", H.dtype)
print(f"H memory: {H.nbytes / 1024**3:.3f} GB")

herm = float(np.max(np.abs(H - H.conj().T)))
assert herm < 1e-10, "Hermiticity gate failed"
lap("build_H")

'''
# Calculating Eigen Values and Eigen Vectors (full, dense)
'''
evals, evecs = np.linalg.eigh(H)
del H
lap("eigh")
n_occ = n_val * n_cells                    # filled states of the window
vbm, cbm = float(evals[n_occ - 1]), float(evals[n_occ])  # find VBM and CBM

print(f"Eigen decomposition done ({LOG['timing_s']['eigh']:.0f} s): spectrum [{evals[0]:.4f}, {evals[-1]:.4f}] eV, "
      f"VBM {vbm:.4f} CBM {cbm:.4f} gap {cbm - vbm:.4f} eV")

'''
# Unfolding: FFT of every eigenvector over the cell labels -> weight w_n(k) on the N x N mesh of k
'''
u = unfold_bands(evals, evecs, selected, path, a=a)
del evecs
print(f"Bands unfolded: {len(u['distance'])} mesh points on the path, {len(u['dot_energy'])} dots")
lap("unfold")

'''
# Bands of the flat sheet straight from k space, same gw and spin_orbit
'''
blocks = hopping_blocks(block_builder=build_H_gw) if gw else hopping_blocks()


def flat_bands(k_frac):
    H_k = bloch_hamiltonian(blocks, k_frac)
    if spin_orbit:
        H_k = add_spin_orbit(H_k, SOC_STRENGTH["Mo"], SOC_STRENGTH["S"])
    return np.linalg.eigvalsh(H_k)


k_line, x_line, ticks = k_path(path, 200, a=a)        # 200 points per segment, so K is point 400
E_flat = flat_bands(k_line)
flat_gap = float(E_flat[400, n_val] - E_flat[400, n_val - 1])

# per state: distance (eV) to the nearest flat band at the k it carries, averaged with its weights
offset = band_offset(u["weights"], evals, flat_bands(uniform_mesh(supercell_side)))

checks = {"side": supercell_side, "n_path_points": len(u["distance"]), "hermiticity_eV": herm,
          "weight_sum_error": float(weight_sum_error(u["weights"])),      # sum_k w = 1 and sum_n w = n_orb
          "offset_median_meV": float(np.median(offset) * 1e3), "offset_max_meV": float(offset.max() * 1e3),
          "VBM_eV": vbm, "CBM_eV": cbm, "gap_eV": cbm - vbm, "flat_gap_at_K_eV": flat_gap}
LOG["unfold"] = checks
print(json.dumps(checks))
lap("flat_bands")

'''
# Save datas
'''
np.savez_compressed(P("unfold.npz"), evals=evals, w=u["weights"].astype(np.float32), bins=u["mesh_index"],
                    x_mesh=u["distance"], x_line=x_line, ticks=ticks, labels=np.array(path_labels), E_ideal=E_flat,
                    a=a, dot_x=u["dot_distance"], dot_E=u["dot_energy"], dot_w=u["dot_weight"], offset=offset,
                    n_orb=n_orb, spin_orbit=spin_orbit, gw=gw)    # the first 13 keys are those of band_unfold.save_unfold
lap("save")

'''
# Figure: FFT weight per level (dots, area = weight) on the bands of H(k) computed directly in k space (lines)
'''
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 13, "axes.linewidth": 1.0,
                     "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300})
RED, NAVY, GRY = "#D1495B", "#2E5A87", "0.55"

fig, ax = plt.subplots(figsize=(5.5, 4.6))
ax.plot(x_line, E_flat, color=NAVY, lw=0.8, zorder=2)
ax.scatter(u["dot_distance"], u["dot_energy"], s=20 * u["dot_weight"], color=RED, linewidths=0, zorder=3)
for xc in ticks[1:-1]:
    ax.axvline(xc, color=GRY, lw=0.6, zorder=1)
ax.set_xlim(ticks[0], ticks[-1])
ax.set_ylim(*E_window)
ax.set_xticks(ticks)
ax.set_xticklabels(path_labels)
ax.set_ylabel("$E$ (eV)")
ax.tick_params(direction="in", top=True, right=True)
ax.scatter([], [], s=20, color=RED, label="real-space $H$, FFT")             # legend entries only
ax.plot([], [], color=NAVY, lw=0.8, label="$H(k)$, direct")
ax.legend(loc="center", fontsize=10)                                          # sits in the gap
fig.tight_layout()
fig.savefig(P("fig_unfolded_bands.png"))
plt.close(fig)
lap("figure")

print(f"Done in {LOG['timing_s']['total']:.0f} s -> {OUT_DIR}")
