#!/usr/bin/env python3
"""find_frames.py — pick the reference and deformed frames of a 260910-style MD run.

Rule (in.mos2au111 protocol): stage 1 relaxes the free-standing MoS2 sheet with the
Au-MoS2 interaction OFF; the line `pair_style hybrid/overlay sw eam lj/cut ...` in
log.lammps switches it ON. So
    reference = the last dumped frame at or before the last thermo step printed
                before that pair_style line   (relaxed, free-standing sheet)
    deformed  = the last dumped frame in positions.dat
Both are read from the files, never assumed; the minimizer's final iteration is always
dumped, so the reference is normally the switch step itself.

Use as a library:
    from find_frames import pick_frames
    ref_step, def_step = pick_frames("/media/yehlab/C/JenTe/260910_MoS2_Au_MD/MoS2_Au_flat")
or from the shell (one line per folder):
    python3 find_frames.py /media/yehlab/C/JenTe/260910_MoS2_Au_MD/MoS2_Au_*
"""
from __future__ import annotations

import os
import re
import sys

SWITCH_PATTERN = "lj/cut"          # the pair_style line that turns the Au-MoS2 interaction on
_THERMO_ROW = re.compile(r"^\s*(\d+)\s+-?\d")   # "   13341  -318957.6  ..." thermo data rows


def switch_step(log_path: str, pattern: str = SWITCH_PATTERN) -> int:
    """Last thermo step printed before the first `pair_style ... <pattern>` line."""
    last = None
    with open(log_path) as f:
        for line in f:
            if line.startswith("pair_style") and pattern in line:
                if last is None:
                    raise ValueError(f"{log_path}: pair_style switch found before any thermo output")
                return last
            m = _THERMO_ROW.match(line)
            if m:
                last = int(m.group(1))
    raise ValueError(f"{log_path}: no 'pair_style ... {pattern}' line — interaction never switched on?")


def dumped_steps(dump_path: str, chunk: int = 1 << 24) -> list[int]:
    """All timesteps present in a LAMMPS text dump, by scanning for 'ITEM: TIMESTEP'.
    Reads the file in binary chunks (fast on 0.6 GB files), keeping an overlap so a
    header split across two chunks is still seen."""
    pat = re.compile(rb"ITEM: TIMESTEP\n(\d+)")
    steps, tail = [], b""
    with open(dump_path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            data = tail + block
            steps.extend(int(m.group(1)) for m in pat.finditer(data))
            tail = data[-64:]                       # longer than any header + number
    # the overlap can report a header twice; keep unique, in file order
    seen, out = set(), []
    for s in steps:
        if s not in seen:
            seen.add(s); out.append(s)
    return out


def run_finished(log_path: str) -> bool:
    with open(log_path) as f:
        return any(line.startswith("Total wall time") for line in f)


def pick_frames(run_dir: str) -> tuple[int, int]:
    """(reference step, deformed step) for one run folder; raises if anything is off."""
    log_path, dump_path = os.path.join(run_dir, "log.lammps"), os.path.join(run_dir, "positions.dat")
    if not run_finished(log_path):
        raise RuntimeError(f"{run_dir}: log.lammps has no 'Total wall time' — the run is not finished")
    sw = switch_step(log_path)
    steps = dumped_steps(dump_path)
    before = [s for s in steps if s <= sw]
    if not before:
        raise ValueError(f"{run_dir}: no dumped frame at or before the switch step {sw}")
    return max(before), max(steps)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print(f"{'folder':32s} {'switch':>8s} {'reference':>10s} {'deformed':>10s} {'frames':>7s}")
    for d in sys.argv[1:]:
        try:
            sw = switch_step(os.path.join(d, "log.lammps"))
            ref, dfm = pick_frames(d)
            n = len(dumped_steps(os.path.join(d, "positions.dat")))
            print(f"{os.path.basename(d.rstrip('/')):32s} {sw:8d} {ref:10d} {dfm:10d} {n:7d}")
        except Exception as e:  # report and continue with the next folder
            print(f"{os.path.basename(d.rstrip('/')):32s} ERROR: {e}")
