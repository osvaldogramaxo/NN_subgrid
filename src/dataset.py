r"""
dataset.py

Data pipeline for the CRMHD-TBNN subgrid-scale closure.

Reads high-resolution HDF5 snapshots from the Aenus solver, applies
relativistic Favre filtering to generate coarse-grained fields, and computes
the exact subgrid-scale stress tensor labels for supervised training.

Dataset layout (per HDF5 file):
    hydro/data : (1, Ny+6, Nx+6, 5) — conserved variables [ρ, E, Sx, Sy, Sz]
    thd/data   : (1, Ny+6, Nx+6, 2) — thermodynamic variables [p, cs]
    Physical domain: indices [3:Nx+3, 3:Ny+3] after stripping 3 ghost cells.

Filtering strategy:
    Relativistic Favre filtering with coarsening factor r=4:
        D = ρ W,  D̄ = F(D),  ũ^μ = F(D u^μ) / D̄
    where W = 1/√(1 - v²) is the Lorentz factor.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#: Adiabatic index for the ideal-gas EOS. This is a *fallback only*: it is
#: rebound at runtime by ``src.config.Config.apply`` from the value resolved
#: for the dataset in use, and dataset metadata outranks the config file. The
#: default is 4/3, matching SRHD data produced by ``src/generate_dns.py``
#: (``dns_khi`` records ``eos_gamma = 1.3333...`` in its own metadata). The
#: legacy ``test1/`` snapshots are Newtonian with 5/3 and carry no metadata,
#: so they need ``--gamma 1.6666666666666667`` or the config value.
#: Never read this without having called ``configure()`` first.
GAMMA_EOS = 4.0 / 3.0
GHOST = 3               # Number of ghost cells on each boundary


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

def read_snapshot(filepath: str | Path) -> dict[str, np.ndarray]:
    """Read a single HDF5 snapshot and return primitive variables on the
    physical domain (ghost cells stripped).

    Returns
    -------
    dict with keys:
        rho : (Ny, Nx) rest-mass density
        vx  : (Ny, Nx) x-velocity
        vy  : (Ny, Nx) y-velocity
        p   : (Ny, Nx) thermal pressure
        t   : float, simulation time
    """
    with h5py.File(filepath, "r") as f:
        t = float(f["Parameters"]["t"][()][0])

        # Snapshots written by src/generate_dns.py carry the primitives directly
        # in a `prim` group: [ρ, v_x, v_y, p]. Prefer it — `hydro` there holds
        # *relativistic* conserved variables (D = ρW, S_i = ρhW²v_i), for which
        # the Newtonian inversion v = S/ρ below would be wrong.
        if "prim" in f:
            prim = f["prim"]["data"][0, GHOST:-GHOST, GHOST:-GHOST, :]  # (Ny, Nx, 4)
            return {
                "rho": prim[..., 0],
                "vx": prim[..., 1],
                "vy": prim[..., 2],
                "p": prim[..., 3],
                "t": t,
            }

        hydro = f["hydro"]["data"][0, GHOST:-GHOST, GHOST:-GHOST, :]  # (Ny, Nx, 5)
        thd = f["thd"]["data"][0, GHOST:-GHOST, GHOST:-GHOST, :]      # (Ny, Nx, 2)

    rho = hydro[..., 0]
    sx = hydro[..., 2]
    sy = hydro[..., 3]

    p = thd[..., 0]

    vx = sx / rho
    vy = sy / rho

    # NOTE: an earlier version rescaled cells with v² >= 1 here, on the theory that
    # the t=10.0 and t=13.0 dumps were corrupt. They are not: this data is Newtonian
    # (Γ = 5/3, verified to machine precision -- see `python -m src.config verify`),
    # and v > 1 is unremarkable in a Newtonian flow. The guard was damaging valid
    # data, so it has been removed. What made W = 1/sqrt(1-v²) blow up was reading
    # Newtonian velocities as relativistic ones, which is a provenance error, not a
    # data defect; `src.config.resolve_gamma` now catches that class of mistake.

    return {"rho": rho, "vx": vx, "vy": vy, "p": p, "t": t}


def load_all_snapshots(data_dir: str | Path) -> list[dict[str, np.ndarray]]:
    """Load all HDF5 snapshots from a directory, sorted by filename (time)."""
    pattern = os.path.join(data_dir, "h*.h5")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No HDF5 files found matching {pattern}")
    return [read_snapshot(f) for f in files]


# ---------------------------------------------------------------------------
# Favre Filtering
# ---------------------------------------------------------------------------

def favre_filter(
    field: torch.Tensor,
    rho: torch.Tensor,
    r: int = 4,
) -> torch.Tensor:
    r"""Apply density-weighted (Favre) box filter with coarsening factor r.

    Computes: \widetilde{f} = \overline{\rho f} / \bar{\rho}

    Parameters
    ----------
    field : (Ny, Nx) tensor — the field to Favre-filter (e.g. v_x)
    rho   : (Ny, Nx) tensor — rest-mass density
    r     : int — coarsening factor (box filter width = r)

    Returns
    -------
    (Ny//r, Nx//r) tensor — Favre-filtered field
    """
    rho_f = rho * field
    # Reshape into r×r blocks and average
    Ny, Nx = rho.shape
    rho_coarse = rho.reshape(Ny // r, r, Nx // r, r).mean(dim=(1, 3))
    rho_f_coarse = rho_f.reshape(Ny // r, r, Nx // r, r).mean(dim=(1, 3))
    return rho_f_coarse / rho_coarse


def box_filter(
    field: torch.Tensor,
    r: int = 4,
) -> torch.Tensor:
    """Apply simple box (volume) filter with coarsening factor r.

    Parameters
    ----------
    field : (Ny, Nx) tensor
    r     : int — coarsening factor

    Returns
    -------
    (Ny//r, Nx//r) tensor — box-filtered field
    """
    Ny, Nx = field.shape
    return field.reshape(Ny // r, r, Nx // r, r).mean(dim=(1, 3))


def relativistic_favre_filter(
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    p: torch.Tensor,
    r: int = 4,
) -> dict[str, torch.Tensor]:
    r"""Apply relativistic Favre filtering to primitive fields.

    Uses D = ρ W as the weighting density, where W is the Lorentz factor.
    Computes:
        D̄  = F(D)
        ρ̄  = F(ρ)
        ṽ_x = F(D v_x) / F(D)
        ṽ_y = F(D v_y) / F(D)
        p̄  = F(p)

    Parameters
    ----------
    rho, vx, vy, p : (Ny, Nx) tensors — high-resolution primitive fields
    r : int — coarsening factor

    Returns
    -------
    dict with coarse-grained fields on (Ny//r, Nx//r) grid:
        rho_bar, vx_tilde, vy_tilde, p_bar, W_bar
    """
    v_sq = vx**2 + vy**2
    # Rescale any numerical overshoots past speed of light (e.g. from DNS Gibbs oscillations)
    v_max_sq = 0.98  # v_max ~ 0.99 c, W_max ~ 7.08
    scale = torch.where(v_sq > v_max_sq, torch.sqrt(v_max_sq / v_sq.clamp(min=1e-15)), torch.ones_like(v_sq))
    vx = vx * scale
    vy = vy * scale
    v_sq = (vx**2 + vy**2).clamp(max=v_max_sq)
    W = 1.0 / torch.sqrt((1.0 - v_sq).clamp(min=1e-12))   # Lorentz factor
    D = rho * W                           # relativistic mass density

    Ny, Nx = rho.shape

    # Coarse-grained quantities
    D_bar = box_filter(D, r)
    rho_bar = box_filter(rho, r)
    p_bar = box_filter(p, r)

    # Favre-filtered velocities (weighted by D = ρW)
    D_vx = D * vx
    D_vy = D * vy
    D_vx_bar = box_filter(D_vx, r)
    D_vy_bar = box_filter(D_vy, r)
    vx_tilde = D_vx_bar / D_bar
    vy_tilde = D_vy_bar / D_bar

    # Rescale coarse velocities to ensure they are strictly subluminal
    v_sq_tilde = vx_tilde**2 + vy_tilde**2
    scale_tilde = torch.where(
        v_sq_tilde > v_max_sq,
        torch.sqrt(v_max_sq / v_sq_tilde.clamp(min=1e-15)),
        torch.ones_like(v_sq_tilde),
    )
    vx_tilde = vx_tilde * scale_tilde
    vy_tilde = vy_tilde * scale_tilde
    v_sq_tilde = (vx_tilde**2 + vy_tilde**2).clamp(max=v_max_sq)
    W_tilde = 1.0 / torch.sqrt((1.0 - v_sq_tilde).clamp(min=1e-12))

    return {
        "rho_bar": rho_bar,
        "vx_tilde": vx_tilde,
        "vy_tilde": vy_tilde,
        "p_bar": p_bar,
        "D_bar": D_bar,
        "W_tilde": W_tilde,
    }


# ---------------------------------------------------------------------------
# Exact SGS Stress Tensor Computation
# ---------------------------------------------------------------------------

def compute_energy_momentum_tensor(
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the relativistic energy-momentum tensor T^{\mu\nu}.

    T^{\mu\nu} = ρ h u^μ u^ν + p η^{\mu\nu}

    where h = 1 + ε + p/ρ is the specific enthalpy,
    ε = p / ((Γ-1) ρ) is the specific internal energy,
    and u^μ = W(1, v_x, v_y, 0) is the four-velocity.

    Parameters
    ----------
    rho, vx, vy, p : (...) tensors — primitive fields

    Returns
    -------
    (..., 4, 4) tensor — T^{μν} in contravariant form
    """
    eps = p / ((GAMMA_EOS - 1.0) * rho)  # specific internal energy
    h = 1.0 + eps + p / rho              # specific enthalpy

    v_sq = vx**2 + vy**2
    v_max_sq = 1.0 - 1e-12
    scale = torch.where(v_sq > v_max_sq, torch.sqrt(v_max_sq / v_sq.clamp(min=1e-15)), torch.ones_like(v_sq))
    vx = vx * scale
    vy = vy * scale
    v_sq = (vx**2 + vy**2).clamp(max=v_max_sq)
    W = 1.0 / torch.sqrt((1.0 - v_sq).clamp(min=1e-12))

    # Four-velocity u^μ = (W, W vx, W vy, 0)
    u = torch.stack([W, W * vx, W * vy, torch.zeros_like(W)], dim=-1)

    # ρ h W² (u^μ u^ν) / W² = ρ h u^μ u^ν
    rho_h = rho * h
    T = rho_h.unsqueeze(-1).unsqueeze(-1) * torch.einsum("...i,...j->...ij", u, u)

    # Add pressure term: p η^{μν}
    eta = torch.zeros(*p.shape, 4, 4, dtype=p.dtype, device=p.device)
    eta[..., 0, 0] = -1.0
    eta[..., 1, 1] = 1.0
    eta[..., 2, 2] = 1.0
    eta[..., 3, 3] = 1.0
    T = T + p.unsqueeze(-1).unsqueeze(-1) * eta

    return T


def compute_sgs_stress(
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    p: torch.Tensor,
    r: int = 4,
    project_to_fluid_frame: bool = True,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    r"""Compute the exact subgrid-scale stress tensor from high-res data.

    τ_SGS^{μν} = \overline{T^{μν}} - T^{μν}(\bar{U})

    If project_to_fluid_frame is True, projects into the comoving fluid rest
    frame using the spatial projector h^\mu_\alpha = \delta^\mu_\alpha + \tilde{u}^\mu \tilde{u}_\alpha:
        \tau_{\perp}^{\mu\nu} = h^\mu_\alpha h^\nu_\beta \tau_{\text{SGS}}^{\alpha\beta}
    ensuring exact mathematical consistency with the fluid rest-frame integrity
    basis where u_\mu E^{(n)\mu\nu} \equiv 0.

    Parameters
    ----------
    rho, vx, vy, p : (Ny, Nx) tensors — high-resolution primitive fields
    r : int — coarsening factor
    project_to_fluid_frame : bool — whether to project into fluid rest frame (default: True)

    Returns
    -------
    coarse_fields : dict — Favre-filtered coarse fields
    tau_sgs       : (Ny//r, Nx//r, 4, 4) tensor — exact SGS stress tensor
    """
    # 1. Compute full T^{μν} on the fine grid
    T_fine = compute_energy_momentum_tensor(rho, vx, vy, p)  # (Ny, Nx, 4, 4)

    # 2. Filter T^{μν} to coarse grid
    Ny, Nx = rho.shape
    Nc_y, Nc_x = Ny // r, Nx // r
    T_bar = T_fine.reshape(Nc_y, r, Nc_x, r, 4, 4).mean(dim=(1, 3))

    # 3. Get Favre-filtered coarse primitives
    coarse = relativistic_favre_filter(rho, vx, vy, p, r)

    # 4. Compute T^{μν} from the coarse (filtered) state
    T_coarse = compute_energy_momentum_tensor(
        coarse["rho_bar"],
        coarse["vx_tilde"],
        coarse["vy_tilde"],
        coarse["p_bar"],
    )

    # 5. SGS stress = filtered fine-grid T - coarse-state T
    tau_sgs = T_bar - T_coarse

    # 6. Option A: Project into fluid rest frame to align with integrity basis
    if project_to_fluid_frame:
        W = coarse["W_tilde"]
        vx_c = coarse["vx_tilde"]
        vy_c = coarse["vy_tilde"]
        u_con = torch.stack([W, W * vx_c, W * vy_c, torch.zeros_like(W)], dim=-1)
        u_cov = torch.stack([-W, W * vx_c, W * vy_c, torch.zeros_like(W)], dim=-1)
        delta = torch.eye(4, dtype=rho.dtype, device=rho.device)
        h_mixed = delta + torch.einsum("...i,...j->...ij", u_con, u_cov)
        tau_sgs = torch.einsum("...ia,...jb,...ab->...ij", h_mixed, h_mixed, tau_sgs)

    return coarse, tau_sgs


def compute_sgs_stress_on_faces(
    rho: torch.Tensor,
    vx: torch.Tensor,
    vy: torch.Tensor,
    p: torch.Tensor,
    r: int = 4,
    project_to_fluid_frame: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Exact SGS stress evaluated on cell faces rather than cell centres.

    The cell-centred stress must be interpolated to faces before its divergence can be
    taken, and any symmetric interpolation annihilates the 2Δ mode — so the closure
    cannot act at the grid scale. Filtering with a box *staggered by r/2 fine cells*
    puts the stress directly on the face, and the divergence becomes a compact
    difference of adjacent faces.

    Rolling the fine field by ``-r/2`` before the standard box filter is exactly
    equivalent to filtering with a box displaced by ``+r/2`` fine cells, which centres
    it on the face between adjacent coarse cells.

    Parameters
    ----------
    rho, vx, vy, p : (Ny, Nx) high-resolution primitive fields
    r : int — coarsening factor; must be even

    Returns
    -------
    tau_x : (Ny//r, Nx//r, 4, 4) — stress at x-faces; index i is the face at i+1/2
    tau_y : (Ny//r, Nx//r, 4, 4) — stress at y-faces; index j is the face at j+1/2
    """
    if r % 2:
        raise ValueError(f"Face staggering needs an even coarsening factor, got r={r}")
    half = r // 2

    def _shifted(dim: int) -> torch.Tensor:
        args = [torch.roll(a, -half, dims=dim) for a in (rho, vx, vy, p)]
        _, tau = compute_sgs_stress(
            *args, r=r, project_to_fluid_frame=project_to_fluid_frame,
        )
        return tau

    return _shifted(-1), _shifted(-2)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SGSDataset(Dataset):
    """Dataset of (coarse fields, SGS stress tensor) pairs for a priori training.

    Each sample is a single spatial point on the coarse grid, providing:
        - Coarse primitive fields (ρ̄, ṽ_x, ṽ_y, p̄)
        - Exact SGS stress tensor τ_SGS^{μν} (4×4 symmetric)

    Parameters
    ----------
    data_dir : str or Path
        Directory containing HDF5 snapshot files.
    coarsening_factor : int
        Downsampling factor (default 4: 1024 → 256).
    time_range : tuple[float, float] | None
        If given, only include snapshots with t in [t_min, t_max].
    dtype : torch.dtype
        Tensor dtype (default float64 for physical accuracy).
    """

    def __init__(
        self,
        data_dir: str | Path,
        coarsening_factor: int = 4,
        time_range: tuple[float, float] | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.r = coarsening_factor
        self.dtype = dtype

        # Load all snapshots
        snapshots = load_all_snapshots(data_dir)

        # Filter by time range if specified
        if time_range is not None:
            t_min, t_max = time_range
            snapshots = [s for s in snapshots if t_min <= s["t"] <= t_max]

        if not snapshots:
            raise ValueError(f"No snapshots in time range {time_range}")

        # Process all snapshots: compute coarse fields and SGS stresses
        all_coarse_fields = []
        all_tau_sgs = []
        self.times = []

        for snap in snapshots:
            rho = torch.tensor(snap["rho"], dtype=dtype)
            vx = torch.tensor(snap["vx"], dtype=dtype)
            vy = torch.tensor(snap["vy"], dtype=dtype)
            p = torch.tensor(snap["p"], dtype=dtype)

            coarse, tau_sgs = compute_sgs_stress(rho, vx, vy, p, self.r)

            # Stack coarse primitives: (Nc_y, Nc_x, 4)
            coarse_prims = torch.stack([
                coarse["rho_bar"],
                coarse["vx_tilde"],
                coarse["vy_tilde"],
                coarse["p_bar"],
            ], dim=-1)

            # Flatten spatial dimensions: each grid point is a sample
            Nc_y, Nc_x = coarse_prims.shape[0], coarse_prims.shape[1]
            all_coarse_fields.append(coarse_prims.reshape(-1, 4))
            all_tau_sgs.append(tau_sgs.reshape(-1, 4, 4))
            self.times.append(snap["t"])

        # Concatenate across all snapshots
        self.coarse_fields = torch.cat(all_coarse_fields, dim=0)  # (N, 4)
        self.tau_sgs = torch.cat(all_tau_sgs, dim=0)              # (N, 4, 4)

        # Compute normalization statistics
        self.field_mean = self.coarse_fields.mean(dim=0)
        self.field_std = self.coarse_fields.std(dim=0).clamp(min=1e-12)
        # Mean Frobenius norm, matching TBNNDataset.tau_scale. These two used
        # to disagree (abs().mean() here, a ~7x smaller number) while neither
        # was used for anything; the τ scale now sets the L_tensor denominator
        # floor, so a single definition matters.
        self.tau_scale = self.tau_sgs.norm(dim=(-2, -1)).mean()

    def __len__(self) -> int:
        return self.coarse_fields.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.coarse_fields[idx], self.tau_sgs[idx]


def create_dataloaders(
    data_dir: str | Path,
    coarsening_factor: int = 4,
    train_time: tuple[float, float] = (0.0, 70.0),
    val_time: tuple[float, float] = (70.0, 85.0),
    test_time: tuple[float, float] = (85.0, 100.0),
    batch_size: int = 1024,
    dtype: torch.dtype = torch.float64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from the snapshot directory.

    Parameters
    ----------
    data_dir : path to directory containing HDF5 files
    coarsening_factor : spatial downsampling factor
    train_time, val_time, test_time : (t_min, t_max) ranges
    batch_size : samples per batch
    dtype : tensor dtype

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_ds = SGSDataset(data_dir, coarsening_factor, train_time, dtype)
    val_ds = SGSDataset(data_dir, coarsening_factor, val_time, dtype)
    test_ds = SGSDataset(data_dir, coarsening_factor, test_time, dtype)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "test1"

    print(f"Loading first snapshot from {data_dir}...")
    snap = read_snapshot(sorted(glob.glob(os.path.join(data_dir, "h*.h5")))[0])
    print(f"  Time: {snap['t']:.2f}")
    print(f"  ρ shape: {snap['rho'].shape}, range: [{snap['rho'].min():.4f}, {snap['rho'].max():.4f}]")
    print(f"  vx range: [{snap['vx'].min():.4f}, {snap['vx'].max():.4f}]")
    print(f"  p range: [{snap['p'].min():.4f}, {snap['p'].max():.4f}]")

    rho = torch.tensor(snap["rho"], dtype=torch.float64)
    vx = torch.tensor(snap["vx"], dtype=torch.float64)
    vy = torch.tensor(snap["vy"], dtype=torch.float64)
    p = torch.tensor(snap["p"], dtype=torch.float64)

    coarse, tau_sgs = compute_sgs_stress(rho, vx, vy, p, r=4)
    print(f"\nCoarse grid shape: {coarse['rho_bar'].shape}")
    print(f"SGS stress tensor shape: {tau_sgs.shape}")
    print(f"  τ_SGS Frobenius norm: {torch.norm(tau_sgs).item():.6e}")
    print(f"  τ_SGS max abs entry:  {tau_sgs.abs().max().item():.6e}")

    # Check symmetry
    asym = (tau_sgs - tau_sgs.transpose(-1, -2)).abs().max().item()
    print(f"  τ_SGS asymmetry:      {asym:.6e}")
