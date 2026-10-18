"""Run the real learned closure on a real GPU, in the solver, in float64.

Two previous submissions died inside LearnedMInITClosure on the cluster while
passing every CPU test, so the fix is verified on the device this time before
another long rollout is queued.
"""
import torch

from src.minit_learned import LearnedMInITClosure
from src.solver import RHDSolver

assert torch.cuda.is_available(), "no CUDA device"
dev = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0),
      "sm_%d%d" % torch.cuda.get_device_capability(0))

N, DT = 64, torch.float64
y = torch.linspace(-0.5, 0.5, N, dtype=DT).unsqueeze(-1)
rho = (1.5 + 0.5 * torch.tanh((0.25 - y.abs()) / 0.05)).expand(N, N).contiguous()
vx = (0.4 * torch.tanh((0.25 - y.abs()) / 0.05)).expand(N, N).contiguous()

solver = RHDSolver(N, N, cfl=0.4, dtype=DT, device=dev)
solver.initialize(rho.to(dev), vx.to(dev), torch.zeros(N, N, dtype=DT, device=dev),
                  torch.full((N, N), 2.5, dtype=DT, device=dev))

ck = "results/minit/minit_coeff_residual.pt"
closure = LearnedMInITClosure.load(ck, solver.dx, solver.dy, device=dev)
print("net dtype:", closure.net.conv1.weight.dtype,
      "device:", closure.net.conv1.weight.device)

tau, alpha = closure(solver.prims)
print("closure call OK: tau", tuple(tau.shape), tau.dtype,
      "finite", bool(torch.isfinite(tau).all()))
print("e_KH mean %.4e max %.4e" % (float(closure.e_KH.mean()), float(closure.e_KH.max())))

solver.evolve(solver.t + 0.05, sgs_closure=closure, verbose=False, progress_bar=False)
print("evolve OK, prims finite:", bool(torch.isfinite(solver.prims).all()))

# CPU/GPU agreement: float64 throughout means these must match to round-off.
cpu = LearnedMInITClosure.load(ck, solver.dx, solver.dy, device="cpu")
tau_cpu, _ = cpu(solver.prims.cpu())
tau_gpu, _ = closure(solver.prims)
d = float((tau_cpu - tau_gpu.cpu()).abs().max() / tau_cpu.abs().max().clamp(min=1e-300))
print(f"max relative CPU-vs-GPU difference: {d:.3e}")
assert d < 1e-10, "float64 path is not reproducing the CPU result"
print("\nALL GPU CHECKS PASSED")
