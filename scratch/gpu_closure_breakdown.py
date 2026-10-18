"""Where does the learned closure's time actually go, and is it unusual?

The closure adds ~620% to a solver step. Before optimising the convolution --
the part that was just changed -- attribute the cost: how many times is the
closure called per step, and what fraction of a call is convolution at all?
The TBNN is the baseline that matters, since rollouts already pay its cost.
"""
import time

import torch

from src.minit import MInITClosure
from src.minit_learned import (
    LearnedMInITClosure, compute_invariants, predict_e_kh,
)
from src.minit import assemble_tau
from src.solver import RHDSolver, compute_dt

dev = torch.device("cuda")
DT = torch.float64
N = 64
print("device:", torch.cuda.get_device_name(0))


def bench(fn, n=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


y = torch.linspace(-0.5, 0.5, N, dtype=DT).unsqueeze(-1)
rho = (1.5 + 0.5 * torch.tanh((0.25 - y.abs()) / 0.05)).expand(N, N).contiguous()
vx = (0.4 * torch.tanh((0.25 - y.abs()) / 0.05)).expand(N, N).contiguous()
solver = RHDSolver(N, N, cfl=0.4, dtype=DT, device=dev)
solver.initialize(rho.to(dev), vx.to(dev), torch.zeros(N, N, dtype=DT, device=dev),
                  torch.full((N, N), 2.5, dtype=DT, device=dev))
prims = solver.prims
dt_step = compute_dt(prims, solver.dx, solver.dy, solver.cfl)

closure = LearnedMInITClosure.load(
    "results/minit/minit_coeff_residual.pt", solver.dx, solver.dy, device=dev)
cfg, net = closure.cfg, closure.net

# --- how many closure calls per solver step? --------------------------------
calls = {"n": 0}
inner = closure.__call__


class Counting:
    def __call__(self, p):
        calls["n"] += 1
        return inner(p)


solver.step(dt_step, sgs_closure=Counting())
print(f"\nclosure calls per solver step: {calls['n']}")

# --- breakdown of one call ---------------------------------------------------
print("\n=== one closure call, 64x64 float64 ===")
with torch.no_grad():
    t_feat = bench(lambda: compute_invariants(
        prims[..., 0], prims[..., 1], prims[..., 2], solver.dx, solver.dy, cfg.lambda_KH))
    feats, base = compute_invariants(
        prims[..., 0], prims[..., 1], prims[..., 2], solver.dx, solver.dy, cfg.lambda_KH)
    t_net = bench(lambda: net(feats))
    t_pred = bench(lambda: predict_e_kh(
        net, cfg, prims[..., 0], prims[..., 1], prims[..., 2], solver.dx, solver.dy))
    e = predict_e_kh(net, cfg, prims[..., 0], prims[..., 1], prims[..., 2],
                     solver.dx, solver.dy)
    t_tau = bench(lambda: assemble_tau(e, prims, closure.coeffs))
    t_all = bench(lambda: closure(prims))

print(f"  compute_invariants (13 periodic_gradient_2d passes) : {t_feat:6.3f} ms")
print(f"  CoefficientNet forward (3 convolutions)             : {t_net:6.3f} ms")
print(f"  predict_e_kh total                                  : {t_pred:6.3f} ms")
print(f"  assemble_tau                                        : {t_tau:6.3f} ms")
print(f"  full closure __call__                               : {t_all:6.3f} ms")
print(f"  -> convolution is {100 * t_net / t_all:.0f}% of a call; "
      f"features are {100 * t_feat / t_all:.0f}%")

# --- baseline: what do the existing closures cost? ---------------------------
print("\n=== step cost by closure, 64x64 float64 ===")
t_unclosed = bench(lambda: solver.step(dt_step), n=20, warmup=5)
print(f"  unclosed            : {t_unclosed:6.2f} ms")

mc = MInITClosure(dx=solver.dx, dy=solver.dy, delta_f=solver.dx, mode="equilibrium")
mc.initialize(prims)
t_minit = bench(lambda: solver.step(dt_step, sgs_closure=mc), n=20, warmup=5)
print(f"  MInIT (equilibrium) : {t_minit:6.2f} ms  ({100*(t_minit-t_unclosed)/t_unclosed:+.0f}%)")

t_learned = bench(lambda: solver.step(dt_step, sgs_closure=closure), n=20, warmup=5)
print(f"  learned coefficient : {t_learned:6.2f} ms  ({100*(t_learned-t_unclosed)/t_unclosed:+.0f}%)")

try:
    from src.evaluate_rollout import TBNNSGSClosure
    from utils import CRMHD_TBNN
    model, ck = CRMHD_TBNN.from_checkpoint(
        "checkpoints_fullrange/best_model.pt", device=dev, dtype=DT)
    model.eval()
    tc = TBNNSGSClosure(model, ck["inv_mean"].to(dev), ck["inv_std"].to(dev),
                        ck["thermo_mean"].to(dev), ck["thermo_std"].to(dev),
                        solver.dx, solver.dy)
    t_tbnn = bench(lambda: solver.step(dt_step, sgs_closure=tc), n=20, warmup=5)
    print(f"  TBNN                : {t_tbnn:6.2f} ms  ({100*(t_tbnn-t_unclosed)/t_unclosed:+.0f}%)")
except Exception as e:
    print(f"  TBNN                : unavailable -> {type(e).__name__}: {str(e)[:100]}")


# --- can the faster conv2d path be had without global state? -----------------
# torch.backends.cudnn.flags() is a context manager, so disabling cuDNN for the
# duration of the forward pass would scope the change instead of mutating the
# process. Worth having only if it actually works on sm_70 and is actually
# faster than the GEMM path.
print("\n=== scoped cuDNN disable (torch.backends.cudnn.flags) ===")
import torch.nn.functional as F
from src.minit_learned import _conv2d_gemm

x = torch.randn(1, 10, 66, 66, device=dev, dtype=DT)
w = torch.randn(32, 10, 3, 3, device=dev, dtype=DT)
b = torch.randn(32, device=dev, dtype=DT)
try:
    with torch.backends.cudnn.flags(enabled=False):
        ref = F.conv2d(x, w, b)
    torch.cuda.synchronize()
    print("  scoped disable WORKS on sm_70")

    def scoped():
        with torch.backends.cudnn.flags(enabled=False):
            return F.conv2d(x, w, b)

    t_scoped = bench(scoped)
    t_gemm = bench(lambda: _conv2d_gemm(x, w, b))
    agree = float((ref - _conv2d_gemm(x, w, b)).abs().max())
    print(f"  scoped conv2d {t_scoped:.3f} ms | gemm {t_gemm:.3f} ms "
          f"| ratio {t_gemm / t_scoped:.2f}x | max abs diff {agree:.2e}")
except Exception as e:
    print(f"  scoped disable FAILS -> {type(e).__name__}: {str(e)[:140]}")
