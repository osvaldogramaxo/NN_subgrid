r"""
plot_dns_gif.py

Build an animation GIF directly from a directory of DNS snapshot HDF5 files
(as produced by src/generate_dns.py), e.g. dns_khi/h00000000.h5 ... h00000060.h5.

Each file holds a single-timestep snapshot with ghost cells; this script trims
the ghost region and animates one primitive field (rho, vx, vy, p) over the
sorted sequence of files.

Usage
-----
    python -m src.plot_dns_gif [dir] [options]

Options
-------
    --field {rho,vx,vy,p}   Field to animate (default: rho)
    --out PATH              Output GIF path (default: <dir>_<field>.gif)
    --fps INT               Playback frames per second (default: 10)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

PRIM_VARS = ["rho", "vx", "vy", "p"]


def _load_frame(path: str, field: str, ghost: int) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as f:
        idx = PRIM_VARS.index(field)
        data = f["prim/data"][0, :, :, idx]
        t = float(f["Parameters/t"][0])
        g = ghost if ghost else int(f["Metadata"].attrs.get("ghost", 0))
    if g:
        data = data[g:-g, g:-g]
    return data, t


def build_gif(dns_dir: str, field: str, out_path: str | None, fps: int) -> str:
    paths = sorted(glob.glob(os.path.join(dns_dir, "h*.h5")))
    if not paths:
        raise ValueError(f"No h*.h5 files found in {dns_dir}")

    with h5py.File(paths[0], "r") as f:
        ghost = int(f["Metadata"].attrs.get("ghost", 0))

    frames, times = [], []
    for p in paths:
        data, t = _load_frame(p, field, ghost)
        frames.append(data)
        times.append(t)

    vmin = min(fr.min() for fr in frames)
    vmax = max(fr.max() for fr in frames)

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    im = ax.imshow(frames[0].T, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.85, label=field)
    title = ax.set_title("")

    def update(i: int):
        im.set_data(frames[i].T)
        title.set_text(f"{field}   t = {times[i]:.4f}   (frame {i + 1}/{len(frames)})")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    if out_path is None:
        out_path = f"{dns_dir.rstrip('/')}_{field}.gif"
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", nargs="?", default="dns_khi", help="Directory of h*.h5 snapshots (default: dns_khi)")
    ap.add_argument("--field", default="rho", choices=PRIM_VARS)
    ap.add_argument("--out", default=None, help="Output GIF path")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    out_path = build_gif(args.dir, args.field, args.out, args.fps)
    print(f"saved {out_path}")


if __name__ == "__main__":
    sys.exit(main())
