"""Example_SpinOrbit_GW.py — bands of the flat MoS2 sheet in four versions of the model, in k space and from a
real-space supercell. Written as LDOS_MoS2_Au_calculation.py and unfold_bands.py are, but no MD run is read (the
sheet carries the one strain set below), so there is one checks.json at the end and no running log.txt.

    MODELBRIDGE=~/Code/ModelBridge python3 Example_SpinOrbit_GW.py [out_dir]      (Mac)
    python3 Example_SpinOrbit_GW.py [out_dir]                                      (sim)

The four versions and the size of H(k): without spin 11x11, with SOC 22x22, GW 11x11, GW + SOC 22x22.
Steps (each is one ModelBridge call): H(k) on Γ-M-K-Γ (hopping_blocks with build_H or build_H_gw,
bloch_hamiltonian, add_spin_orbit) -> eigvalsh -> first figure. Then a periodic N x N supercell of each version
(build_full_H or build_full_H_gw, add_spin_orbit_real_space) -> full eigh -> FFT of every eigenvector over the
cells (unfold_bands) -> second figure: the real-space solution (dots, area = weight) on the k-space bands (lines).

Output in OUT_DIR (default: bands_spin_orbit_gw next to this script): bands_spin_orbit_gw.png,
bands_real_space_unfolded.png, bands.npz (the plotted arrays), checks.json. About 3 minutes, 2.7 GB.
`Library/` comes from the ModelBridge clone in MODELBRIDGE (below);
Library/Bloch.py, SpinOrbit.py, GW.py and Unfold.py must be in it.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# let the script find the source code
MODELBRIDGE = os.environ.get("MODELBRIDGE", "/media/yehlab/C/JenTe/ModelBridge")
sys.path.insert(0, MODELBRIDGE)

from Library.Hamiltonian import build_full_H                      # noqa: E402
from Library.Bloch import hopping_blocks, bloch_hamiltonian, uniform_mesh, k_path   # noqa: E402
from Library.SpinOrbit import add_spin_orbit, add_spin_orbit_real_space, SOC_STRENGTH  # noqa: E402
from Library.GW import build_H_gw, build_full_H_gw                # noqa: E402
from Library.Unfold import unfold_bands, band_offset, weight_sum_error   # noqa: E402

t0 = time.time()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "bands_spin_orbit_gw")
strain = {"exx": 0.0, "eyy": 0.0, "exy": 0.0}     # uniform strain of the sheet
lambda_Mo = SOC_STRENGTH["Mo"]           # spin-orbit strength of Mo (eV), Fang 2015 Table VIII
lambda_S = SOC_STRENGTH["S"]             # of S
supercell_side = 15                      # real space: N x N cells, periodic; K is on the k mesh when 3 divides N
path = ["G", "M", "K", "G"]              # corners of the k path, names of Library.Bloch.HIGH_SYMMETRY
path_labels = ["Γ", "M", "K", "Γ"]       # the same corners as written on the figures
E_window = (-3.0, 4.5)                   # energy range of the figures (eV), from the valence top at K

# the four versions of the model: title of the panel, spin-orbit coupling, GW correction
MODELS = [("without spin 11x11", False, False),
          ("with SOC 22x22",     True,  False),
          ("GW 11x11",           False, True),
          ("GW + SOC 22x22",     True,  True)]

os.makedirs(OUT_DIR, exist_ok=True)
P = lambda s: os.path.join(OUT_DIR, s)                            # noqa: E731
checks = {}


def H_of_k(k_frac, soc, gw):
    blocks = hopping_blocks(strain, block_builder=build_H_gw) if gw else hopping_blocks(strain)
    H_k = bloch_hamiltonian(blocks, k_frac)
    if soc:
        H_k = add_spin_orbit(H_k, lambda_Mo, lambda_S)
    return H_k


'''
# k space: bands of each model along the path, and on the k mesh the supercell allows
'''
k_line, x_line, ticks = k_path(path, 200)              # 200 points per segment, so K is point 400
mesh = uniform_mesh(supercell_side)

E_line, E_mesh, E_top = {}, {}, {}
for title, soc, gw in MODELS:
    n_val = 14 if soc else 7                           # valence bands
    E_line[title] = np.linalg.eigvalsh(H_of_k(k_line, soc, gw))
    E_mesh[title] = np.linalg.eigvalsh(H_of_k(mesh, soc, gw))
    E_top[title] = E_line[title][400, n_val - 1]       # valence top at K
    checks[title] = {"H_k_size": E_line[title].shape[1],
                     "gap_at_K_eV": float(E_line[title][400, n_val] - E_top[title])}
    print(f"{title:19s}: gap at K {checks[title]['gap_at_K_eV']:.4f} eV")

E_soc = E_line["with SOC 22x22"][400]
checks["valence_splitting_at_K_meV"] = float(1e3 * (E_soc[13] - E_soc[12]))
checks["conduction_splitting_at_K_meV"] = float(1e3 * (E_soc[15] - E_soc[14]))
print(f"spin splitting at K: valence {checks['valence_splitting_at_K_meV']:.1f} meV, "
      f"conduction {checks['conduction_splitting_at_K_meV']:.1f} meV")

'''
# Real space: the same four models on a flat periodic supercell, unfolded to k space by FFT
'''
cell_x, cell_y = np.indices((supercell_side, supercell_side)).reshape(2, -1)
n_cells = supercell_side ** 2
cells = pd.DataFrame({"mo_id": np.arange(n_cells), "cell_x": cell_x, "cell_y": cell_y, "cell": np.arange(n_cells)})
strain_df = pd.DataFrame({"id": np.arange(n_cells), "exx": strain["exx"], "eyy": strain["eyy"], "exy": strain["exy"]})

dots = {}
for title, soc, gw in MODELS:
    H = build_full_H_gw(cells, strain_df) if gw else build_full_H(cells, strain_df)
    if soc:
        H = add_spin_orbit_real_space(H, lambda_Mo, lambda_S)      # complex, 22 per cell
    else:
        # without spin H is real symmetric. Keep it real: eigh then uses the real solver
        if np.abs(H.imag).max() != 0:
            raise ValueError("H has a nonzero imaginary part; keep it complex")
        H = np.ascontiguousarray(H.real)

    herm = float(np.max(np.abs(H - H.conj().T)))
    assert herm < 1e-10, "Hermiticity gate failed"

    evals, evecs = np.linalg.eigh(H)
    dots[title] = unfold_bands(evals, evecs, cells, path)

    # per state: distance (eV) to the nearest k-space band at the k it carries; zero on a flat supercell
    offset = band_offset(dots[title]["weights"], evals, E_mesh[title])
    checks[title].update({"H_size": H.shape[0], "hermiticity_eV": herm,
                          "real_space_against_k_space_eV": float(offset.max()),
                          "weight_sum_error": float(weight_sum_error(dots[title]["weights"]))})
    print(f"{title:19s}: H {H.shape}, real space against k space {offset.max():.1e} eV")

'''
# Save datas
'''
data = {"x_line": x_line, "ticks": ticks, "labels": np.array(path_labels), "titles": np.array([m[0] for m in MODELS])}
for i, (title, soc, gw) in enumerate(MODELS):
    data[f"E_line_{i}"] = E_line[title] - E_top[title]
    data[f"dot_x_{i}"] = dots[title]["dot_distance"]
    data[f"dot_E_{i}"] = dots[title]["dot_energy"] - E_top[title]
    data[f"dot_w_{i}"] = dots[title]["dot_weight"]
np.savez_compressed(P("bands.npz"), **data)

checks["time_s"] = round(time.time() - t0, 1)
json.dump(checks, open(P("checks.json"), "w"), indent=1)
print(json.dumps(checks))

'''
# Figures: the bands in k space; then the real-space dots (area = weight) on the k-space lines
'''
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 12, "axes.labelsize": 13, "axes.linewidth": 1.0,
                     "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300})
RED, NAVY, GRY = "#D1495B", "#2E5A87", "0.55"

for name, with_dots in (("bands_spin_orbit_gw.png", False), ("bands_real_space_unfolded.png", True)):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)

    for ax, (title, soc, gw) in zip(axes.flat, MODELS):
        ax.plot(x_line, E_line[title] - E_top[title], color=NAVY, lw=0.8, zorder=2)
        if with_dots:
            u = dots[title]
            ax.scatter(u["dot_distance"], u["dot_energy"] - E_top[title], s=20 * u["dot_weight"], color=RED,
                       linewidths=0, zorder=3)
        for xc in ticks[1:-1]:
            ax.axvline(xc, color=GRY, lw=0.6, zorder=1)
        ax.set_xlim(ticks[0], ticks[-1])
        ax.set_ylim(*E_window)
        ax.set_xticks(ticks)
        ax.set_xticklabels(path_labels)
        ax.set_title(title)
        ax.tick_params(direction="in", top=True, right=True)

    for ax in axes[:, 0]:
        ax.set_ylabel("$E - E_v(\\mathrm{K})$ (eV)")
    if with_dots:
        axes[0, 0].scatter([], [], s=20, color=RED, label=f"real-space $H$ {supercell_side}x{supercell_side}, FFT")
        axes[0, 0].plot([], [], color=NAVY, lw=0.8, label="$H(k)$, direct")                # legend entries only
        axes[0, 0].legend(loc="center", fontsize=10)                                       # sits in the gap
    fig.tight_layout()
    fig.savefig(P(name))
    plt.close(fig)

print(f"Done in {time.time() - t0:.0f} s -> {OUT_DIR}")
