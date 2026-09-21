r"""
plot_rollout_gif.py

Build a comparison GIF from a rollout_trajectory.pt saved by evaluate_rollout.py
(--save-history / --make-anim).

Each panel (dns / tbnn / exact / unclosed) may have recorded a different number of
frames if its solver diverged partway through the run: this script does not assume a
shared frame count. It walks a master timeline built from the union of all panels'
recorded times, and for any panel that stopped early it holds on the last valid frame
and stamps it "DIVERGED @ t=...", using the per-panel divergence time saved by
evaluate_rollout.py's FloatingPointError handling.

Usage
-----
    python -m src.plot_rollout_gif [path/to/rollout_trajectory.pt] [options]

Options
-------
    --field {rho,vx,vy,p}   Field to animate (default: rho)
    --out PATH              Output GIF path (default: <traj>_<field>.gif)
    --fps INT                Playback frames per second (default: 8)
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

PANEL_ORDER = ["exact", "dns", "tbnn", "unclosed"]
PANEL_TITLES = {
    "exact": "Exact DNS-Closed",
    "dns": "Filtered DNS",
    "tbnn": "TBNN-Closed",
    "unclosed": "Unclosed (no SGS)",
}


def _as_numpy(x) -> np.ndarray:
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def build_gif(traj_path: str, field: str, out_path: str | None, fps: int) -> str:
    d = torch.load(traj_path, map_location="cpu", weights_only=False)
    diverged_at: dict[str, float] = d.get("diverged_at", {}) or {}

    panels = [p for p in PANEL_ORDER if p in d and d[p] and field in d[p]]
    if not panels:
        raise ValueError(f"No panel in {traj_path} has field '{field}'")

    panel_data = {p: _as_numpy(d[p][field]) for p in panels}
    panel_times = {p: _as_numpy(d[p]["t"]) for p in panels}

    # Master timeline: union of every panel's recorded times, so the animation still
    # advances even where one solver stopped recording early (divergence) or has more
    # frames than another.
    all_times = np.unique(np.concatenate([panel_times[p] for p in panels]))
    n_frames = len(all_times)

    # For each panel, map master-timeline index -> latest panel frame index whose time
    # is <= the master time. This holds each panel on its last known state once its
    # own recording stops.
    panel_frame_idx = {}
    for p in panels:
        pt = panel_times[p]
        idx = np.searchsorted(pt, all_times, side="right") - 1
        panel_frame_idx[p] = np.clip(idx, 0, len(pt) - 1)

    vmin = min(panel_data[p].min() for p in panels)
    vmax = max(panel_data[p].max() for p in panels)

    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.2 * len(panels), 4.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    ims = []
    div_texts = []
    for ax, p in zip(axes, panels):
        im = ax.imshow(panel_data[p][0], origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(PANEL_TITLES.get(p, p))
        ax.set_xticks([])
        ax.set_yticks([])
        ims.append(im)
        txt = ax.text(
            0.5, -0.06, "", transform=ax.transAxes, ha="center", va="top",
            color="crimson", fontsize=9, fontweight="bold",
        )
        div_texts.append(txt)

    fig.colorbar(ims[0], ax=list(axes), shrink=0.8, label=field)
    suptitle = fig.suptitle("")

    last_idx = {p: len(panel_times[p]) - 1 for p in panels}

    def update(i: int):
        t_master = all_times[i]
        for im, txt, p in zip(ims, div_texts, panels):
            fi = panel_frame_idx[p][i]
            im.set_data(panel_data[p][fi])
            stalled = fi == last_idx[p] and t_master > panel_times[p][fi] + 1e-9
            if p in diverged_at and stalled:
                txt.set_text(f"DIVERGED @ t={diverged_at[p]:.3f}")
            else:
                txt.set_text("")
        suptitle.set_text(f"t = {t_master:.4f}   (frame {i + 1}/{n_frames})")
        return ims + div_texts + [suptitle]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    if out_path is None:
        out_path = traj_path.rsplit(".", 1)[0] + f"_{field}.gif"
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traj", nargs="?", default="results/rollout_trajectory.pt",
                     help="Path to rollout_trajectory.pt (default: results/rollout_trajectory.pt)")
    ap.add_argument("--field", default="rho", choices=["rho", "vx", "vy", "p"])
    ap.add_argument("--out", default=None, help="Output GIF path")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    out_path = build_gif(args.traj, args.field, args.out, args.fps)
    print(f"saved {out_path}")

    d = torch.load(args.traj, map_location="cpu", weights_only=False)
    diverged_at = d.get("diverged_at", {})
    if diverged_at:
        print(f"divergence times: {diverged_at}")
    else:
        print("no divergence recorded in this trajectory file")


if __name__ == "__main__":
    sys.exit(main())
