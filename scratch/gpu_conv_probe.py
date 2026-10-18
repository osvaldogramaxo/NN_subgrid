"""Which convolution paths actually execute on this cluster's GPUs?

A double-precision conv2d on CUDA was assumed to be the cause of
"GET was unable to find an engine to execute this computation". Forcing float32
did not fix it, so the assumption was wrong. This enumerates the candidates
rather than guessing a third time.
"""
import torch
import torch.nn.functional as F

print("torch", torch.__version__, "cuda", torch.version.cuda)
try:
    print("cudnn version:", torch.backends.cudnn.version())
except Exception as e:
    # This is the finding, not an error: cuDNN 9.24 refuses SM < 7.5, so on a
    # V100 (SM 7.0) *every* cuDNN path is dead regardless of dtype.
    print("cudnn version: UNAVAILABLE ->", type(e).__name__, str(e)[:160])
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("no CUDA visible")
print("device:", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print(f"compute capability: sm_{cap[0]}{cap[1]}")


def probe(name, fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        print(f"  OK    {name}  -> {tuple(out.shape)} {out.dtype}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}  -> {type(e).__name__}: {str(e)[:120]}")
        return False


for dt in (torch.float32, torch.float64):
    x = torch.randn(1, 10, 64, 64, device="cuda", dtype=dt)
    w = torch.randn(32, 10, 3, 3, device="cuda", dtype=dt)
    b = torch.randn(32, device="cuda", dtype=dt)
    print(f"\n--- {dt} ---")
    probe("conv2d (cudnn on)", lambda: F.conv2d(x, w, b, padding=1))

    torch.backends.cudnn.enabled = False
    probe("conv2d (cudnn OFF)", lambda: F.conv2d(x, w, b, padding=1))
    torch.backends.cudnn.enabled = True

    def gemm():
        xp = F.pad(x, (1, 1, 1, 1), mode="circular")
        u = F.unfold(xp, kernel_size=3)                      # (B, Cin*9, L)
        y = w.reshape(w.shape[0], -1) @ u                    # (B, Cout, L)
        return (y + b[None, :, None]).reshape(1, 32, 64, 64)

    probe("unfold + matmul (cuBLAS)", gemm)
    probe("matmul only", lambda: x.reshape(1, 10, -1).transpose(1, 2) @ torch.randn(10, 32, device="cuda", dtype=dt))
