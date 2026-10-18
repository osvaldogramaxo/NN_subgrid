"""What does bypassing cuDNN actually cost?

_conv2d_gemm (im2col + cuBLAS) replaced F.conv2d because cuDNN 9.24 refuses
sm_70. The honest comparison is not against cuDNN -- that path does not run here
at all -- but against the only other working option, F.conv2d with cuDNN
disabled. The number that matters, though, is the closure's share of a solver
step: a 2x slower convolution is irrelevant if convolution is 1% of the step.
"""
import time

import torch
import torch.nn.functional as F

from src.minit_learned import CoefficientNet, LearnedMInITClosure, _conv2d_gemm
from src.solver import RHDSolver, compute_dt

dev = torch.device("cuda")
print("device:", torch.cuda.get_device_name(0),
      "sm_%d%d" % torch.cuda.get_device_capability(0))


def bench(fn, n=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3          # ms


print("\n=== single 3x3 conv, 10->32 channels, cuDNN disabled for the F.conv2d row ===")
print(f"{'grid':>8} {'dtype':>9} {'gemm ms':>9} {'conv2d ms':>10} {'ratio':>7}")
for N in (64, 128, 256):
    for dt in (torch.float32, torch.float64):
        x = torch.randn(1, 10, N + 2, N + 2, device=dev, dtype=dt)
        w = torch.randn(32, 10, 3, 3, device=dev, dtype=dt)
        b = torch.randn(32, device=dev, dtype=dt)
        t_gemm = bench(lambda: _conv2d_gemm(x, w, b))
        torch.backends.cudnn.enabled = False
        t_conv = bench(lambda: F.conv2d(x, w, b))
        torch.backends.cudnn.enabled = True
        print(f"{N:>8} {str(dt).replace('torch.',''):>9} {t_gemm:>9.3f} "
              f"{t_conv:>10.3f} {t_gemm / t_conv:>7.2f}x")

print("\n=== full CoefficientNet forward (the thing the solver actually calls) ===")
print(f"{'grid':>8} {'dtype':>9} {'ms':>9}")
for N in (64, 128, 256):
    for dt in (torch.float32, torch.float64):
        net = CoefficientNet().to(device=dev, dtype=dt)
        inp = torch.randn(1, N, N, 10, device=dev, dtype=dt)
        with torch.no_grad():
            print(f"{N:>8} {str(dt).replace('torch.',''):>9} "
                  f"{bench(lambda: net(inp)):>9.3f}")

print("\n=== closure cost vs a full solver step, float64, 64x64 (the r=16 case) ===")
DT = torch.float64
for N in (64, 128):
    y = torch.linspace(-0.5, 0.5, N, dtype=DT).unsqueeze(-1)
    rho = (1.5 + 0.5 * torch.tanh((0.25 - y.abs()) / 0.05)).expand(N, N).contiguous()
    vx = (0.4 * torch.tanh((0.25 - y.abs()) / 0.05)).expand(N, N).contiguous()
    solver = RHDSolver(N, N, cfl=0.4, dtype=DT, device=dev)
    solver.initialize(rho.to(dev), vx.to(dev),
                      torch.zeros(N, N, dtype=DT, device=dev),
                      torch.full((N, N), 2.5, dtype=DT, device=dev))
    closure = LearnedMInITClosure.load(
        "results/minit/minit_coeff_residual.pt", solver.dx, solver.dy, device=dev)

    t_closure = bench(lambda: closure(solver.prims), n=30)
    dt_step = compute_dt(solver.prims, solver.dx, solver.dy, solver.cfl)
    t_unclosed = bench(lambda: solver.step(dt_step), n=20, warmup=5)
    t_closed = bench(lambda: solver.step(dt_step, sgs_closure=closure), n=20, warmup=5)
    print(f"  {N}x{N}: closure call {t_closure:.3f} ms | "
          f"step unclosed {t_unclosed:.2f} ms | step closed {t_closed:.2f} ms | "
          f"overhead {100 * (t_closed - t_unclosed) / t_unclosed:+.1f}%")
