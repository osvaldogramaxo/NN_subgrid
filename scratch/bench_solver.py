"""Profile the RHD solver hot path. Run under gpurun (5-minute cap)."""
import sys, time, argparse
sys.path.insert(0, ".")
import torch
import src.solver as S
from src.generate_dns import khi_initial_conditions

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1024)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--dtype", default="float64")
args = ap.parse_args()

S.set_gamma_eos(4/3)
dev = torch.device("cuda")
dt_t = torch.float64 if args.dtype == "float64" else torch.float32
N = args.n

rho, vx, vy, p, _, _ = khi_initial_conditions(
    N, N, v_shear=0.8, thickness=0.02, rho1=1.0, rho2=2.0, pressure=2.5,
    seed_amplitude=0.01, seed_kmin=4, seed_kmax=32, seed_envelope_width=None, rng_seed=0)
sol = S.RHDSolver(Ny=N, Nx=N, cfl=0.4, reconstruction="mp5", riemann_solver="hllc",
                  dtype=dt_t, device=dev)
sol.initialize(*(torch.from_numpy(a) for a in (rho, vx, vy, p)))

def sync(): torch.cuda.synchronize()

def timeit(fn, n, warmup=3):
    for _ in range(warmup): fn()
    sync(); t0 = time.perf_counter()
    for _ in range(n): fn()
    sync(); return (time.perf_counter() - t0) / n * 1e3   # ms

print(f"=== {N}x{N} {args.dtype} on {torch.cuda.get_device_name(0)} ===", flush=True)

# --- whole step ---------------------------------------------------------
dt = S.compute_dt(sol.prims, sol.dx, sol.dy, sol.cfl)
t_step = timeit(lambda: sol.step(dt), args.steps)
print(f"full step                     {t_step:8.2f} ms", flush=True)

# --- components ---------------------------------------------------------
prims = sol.prims
U = S.primitives_to_conserved(prims[...,0], prims[...,1], prims[...,2], prims[...,3])
t_dt   = timeit(lambda: S.compute_dt(prims, sol.dx, sol.dy, sol.cfl), 50)
t_p2c  = timeit(lambda: S.primitives_to_conserved(prims[...,0],prims[...,1],prims[...,2],prims[...,3]), 50)
t_c2p  = timeit(lambda: S.conserved_to_primitives(U), 20)
t_rhs  = timeit(lambda: S.compute_rhs(prims, sol.dx, sol.dy, None, "mp5", "hllc", True), 20)
print(f"  compute_dt                  {t_dt:8.2f} ms   ({100*t_dt/t_step:4.1f}% of step)")
print(f"  primitives_to_conserved     {t_p2c:8.2f} ms")
print(f"  conserved_to_primitives     {t_c2p:8.2f} ms   ({100*3*t_c2p/t_step:4.1f}% of step, x3 stages)")
print(f"  compute_rhs                 {t_rhs:8.2f} ms   ({100*3*t_rhs/t_step:4.1f}% of step, x3 stages)", flush=True)

# --- how many Newton iterations are actually used? ----------------------
D = U[...,0].clamp(min=1e-15); tau = U[...,3].clamp(min=1e-15)
gm1 = S.GAMMA_EOS - 1.0
p_it = (gm1*tau/S.GAMMA_EOS).clamp(min=1e-15).clone()
Sx, Sy = U[...,1], U[...,2]; S_sq = Sx**2+Sy**2
for it in range(1, 31):
    z = (tau+D+p_it).clamp(min=1e-15)
    v_sq = (S_sq/z**2).clamp(max=1.0-1e-12)
    W = 1.0/torch.sqrt(1.0-v_sq); rho_i = (D/W).clamp(min=1e-15)
    eps = ((tau+D*(1.0-W)+p_it*(1.0-W**2))/(D*W)).clamp(min=1e-15)
    f = p_it - gm1*rho_i*eps
    h = z/(D*W); dfdp = 1.0-gm1*v_sq*(1.0-1.0/h)
    dp = torch.where(torch.isfinite(-f/dfdp), (-f/dfdp).clamp(-0.8*p_it, 5.0*p_it), torch.zeros_like(p_it))
    p_it = (p_it+dp).clamp(min=1e-15)
    if bool((dp.abs() <= 1e-12*p_it).all()):
        break
print(f"\nNewton iterations to converge (cold start): {it}", flush=True)

# --- cost of the per-iteration host sync --------------------------------
def c2p_nosync(U, iters):
    D = U[...,0].clamp(min=1e-15); tau = U[...,3].clamp(min=1e-15)
    Sx, Sy = U[...,1], U[...,2]; S_sq = Sx**2+Sy**2
    p_ = (gm1*tau/S.GAMMA_EOS).clamp(min=1e-15).clone()
    for _ in range(iters):
        z = (tau+D+p_).clamp(min=1e-15)
        v_sq = (S_sq/z**2).clamp(max=1.0-1e-12)
        W = 1.0/torch.sqrt(1.0-v_sq); r = (D/W).clamp(min=1e-15)
        eps = ((tau+D*(1.0-W)+p_*(1.0-W**2))/(D*W)).clamp(min=1e-15)
        f = p_ - gm1*r*eps
        h = z/(D*W); dfdp = 1.0-gm1*v_sq*(1.0-1.0/h)
        q = -f/dfdp
        dp = torch.where(torch.isfinite(q), q.clamp(-0.8*p_, 5.0*p_), torch.zeros_like(p_))
        p_ = (p_+dp).clamp(min=1e-15)
    return p_
t_ns = timeit(lambda: c2p_nosync(U, it), 20)
print(f"C2P, {it} iters, no sync check   {t_ns:8.2f} ms   vs {t_c2p:.2f} ms with per-iter .all()")
print(f"  -> sync overhead              {t_c2p-t_ns:8.2f} ms  ({100*(t_c2p-t_ns)/t_c2p:.0f}% of C2P)", flush=True)
