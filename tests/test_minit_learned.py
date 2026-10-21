"""Tests for the state-dependent MInIT coefficient closure."""

from __future__ import annotations

import math

import pytest
import torch.nn.functional as F
import torch

from src.minit import assemble_tau, subgrid_energy_from_stress
from src.minit_learned import (
    E_FLOOR,
    INVARIANT_NAMES,
    N_INVARIANTS,
    CoefficientNet,
    LearnedMInITClosure,
    LearnedMInITConfig,
    compute_invariants,
    predict_e_kh,
)

DT = torch.float64


def _prims(Ny=16, Nx=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.empty(Ny, Nx, 4, dtype=DT)
    p[..., 0] = 0.5 + torch.rand(Ny, Nx, generator=g, dtype=DT)      # rho > 0
    p[..., 1] = 0.4 * (torch.rand(Ny, Nx, generator=g, dtype=DT) - 0.5)
    p[..., 2] = 0.4 * (torch.rand(Ny, Nx, generator=g, dtype=DT) - 0.5)
    p[..., 3] = 1.0 + torch.rand(Ny, Nx, generator=g, dtype=DT)      # p > 0
    return p


def _net(width=8, seed=0):
    torch.manual_seed(seed)
    return CoefficientNet(width=width).to(dtype=DT)


def _cfg(**kw):
    base = dict(lambda_KH=15.8 / 64, beta_xx=1.66, beta_yy=0.34, beta_xy=0.012)
    base.update(kw)
    return LearnedMInITConfig(**base)


class TestInvariants:
    def test_shape_and_names_agree(self):
        p = _prims()
        f, base = compute_invariants(p[..., 0], p[..., 1], p[..., 2],
                                     1 / 16, 1 / 16, 0.25)
        assert f.shape == (16, 16, N_INVARIANTS)
        assert len(INVARIANT_NAMES) == N_INVARIANTS
        assert base.shape == (16, 16)

    def test_all_finite(self):
        p = _prims()
        f, base = compute_invariants(p[..., 0], p[..., 1], p[..., 2],
                                     1 / 16, 1 / 16, 0.25)
        assert torch.isfinite(f).all()
        assert torch.isfinite(base).all()

    def test_uniform_flow_has_zero_strain(self):
        """No gradients => no Smagorinsky base, so no sub-grid energy anywhere."""
        rho = torch.full((16, 16), 1.3, dtype=DT)
        vx = torch.full((16, 16), 0.2, dtype=DT)
        vy = torch.zeros(16, 16, dtype=DT)
        f, base = compute_invariants(rho, vx, vy, 1 / 16, 1 / 16, 0.25)
        assert torch.allclose(base, torch.zeros_like(base), atol=1e-24)

    def test_base_is_rho_lambda2_strain2(self):
        p = _prims()
        lam = 0.25
        f, base = compute_invariants(p[..., 0], p[..., 1], p[..., 2],
                                     1 / 16, 1 / 16, lam)
        strain = f[..., INVARIANT_NAMES.index("strain")]
        expect = p[..., 0].clamp(min=1e-12) * lam**2 * strain**2
        assert torch.allclose(base, expect)

    def test_batched_matches_single(self):
        """The training path feeds (T, Ny, Nx); the solver feeds (Ny, Nx)."""
        p = torch.stack([_prims(seed=s) for s in range(3)])
        fb, bb = compute_invariants(p[..., 0], p[..., 1], p[..., 2],
                                    1 / 16, 1 / 16, 0.25)
        for i in range(3):
            fs, bs = compute_invariants(p[i, ..., 0], p[i, ..., 1], p[i, ..., 2],
                                        1 / 16, 1 / 16, 0.25)
            assert torch.allclose(fb[i], fs)
            assert torch.allclose(bb[i], bs)

    def test_translation_equivariance(self):
        """Periodic features must commute with a roll, or the padding is wrong."""
        p = _prims()
        f, _ = compute_invariants(p[..., 0], p[..., 1], p[..., 2], 1 / 16, 1 / 16, 0.25)
        pr = torch.roll(p, shifts=(3, 5), dims=(0, 1))
        fr, _ = compute_invariants(pr[..., 0], pr[..., 1], pr[..., 2],
                                   1 / 16, 1 / 16, 0.25)
        assert torch.allclose(torch.roll(f, shifts=(3, 5), dims=(0, 1)), fr, atol=1e-12)


class TestCoefficientNet:
    def test_receptive_field_is_5x5(self):
        """A one-cell input perturbation must reach exactly the 5x5 block."""
        net = _net()
        net.eval()
        x = torch.zeros(1, 9, 9, N_INVARIANTS, dtype=DT)
        a = net(x)
        x2 = x.clone()
        x2[0, 4, 4, :] = 1.0
        b = net(x2)
        touched = (a - b).abs()[0] > 1e-14
        ii, jj = torch.meshgrid(torch.arange(9), torch.arange(9), indexing="ij")
        inside = (ii - 4).abs() <= 2
        inside &= (jj - 4).abs() <= 2
        assert touched[inside].all(), "5x5 block not fully reached"
        assert not touched[~inside].any(), "influence leaked outside 5x5"

    def test_shape_squeeze_and_batch(self):
        net = _net()
        assert net(torch.zeros(16, 16, N_INVARIANTS, dtype=DT)).shape == (16, 16)
        assert net(torch.zeros(3, 16, 16, N_INVARIANTS, dtype=DT)).shape == (3, 16, 16)

    def test_output_is_bounded(self):
        """out_clamp must hold even for absurd inputs, or exp() destroys a rollout."""
        net = _net()
        wild = 1e6 * torch.randn(8, 8, N_INVARIANTS, dtype=DT)
        out = net(wild)
        assert torch.isfinite(out).all()
        assert out.abs().max() <= net.out_clamp + 1e-9

    def test_circular_padding_no_edge_artifact(self):
        """A rolled input must give a rolled output: catches zero-padding."""
        net = _net()
        x = torch.randn(12, 12, N_INVARIANTS, dtype=DT)
        out = net(x)
        rolled = net(torch.roll(x, shifts=(2, -3), dims=(0, 1)))
        assert torch.allclose(torch.roll(out, shifts=(2, -3), dims=(0, 1)),
                              rolled, atol=1e-12)

    def test_normalization_buffers_persist(self):
        net = _net()
        mean = torch.arange(N_INVARIANTS, dtype=DT)
        std = torch.full((N_INVARIANTS,), 2.0, dtype=DT)
        net.set_normalization(mean, std)
        other = _net(seed=1)
        other.load_state_dict(net.state_dict())
        assert torch.allclose(other.feat_mean, mean)
        assert torch.allclose(other.feat_std, std)

    def test_normalization_clamps_zero_std(self):
        """A constant feature has std 0; dividing by it would produce NaN."""
        net = _net()
        net.set_normalization(torch.zeros(N_INVARIANTS, dtype=DT),
                              torch.zeros(N_INVARIANTS, dtype=DT))
        assert (net.feat_std > 0).all()
        assert torch.isfinite(net(torch.randn(8, 8, N_INVARIANTS, dtype=DT))).all()


class TestPredictEKH:
    def test_non_negative(self):
        p = _prims()
        e = predict_e_kh(_net(), _cfg(), p[..., 0], p[..., 1], p[..., 2], 1 / 16, 1 / 16)
        assert (e >= 0).all()
        assert torch.isfinite(e).all()

    def test_residual_mode_vanishes_with_strain(self):
        """Uniform flow has no strain, so the residual form must give e_KH = 0."""
        rho = torch.full((16, 16), 1.3, dtype=DT)
        vx = torch.full((16, 16), 0.2, dtype=DT)
        vy = torch.zeros(16, 16, dtype=DT)
        e = predict_e_kh(_net(), _cfg(output_mode="residual"),
                         rho, vx, vy, 1 / 16, 1 / 16)
        assert torch.allclose(e, torch.zeros_like(e), atol=1e-12)

    def test_direct_mode_need_not_vanish(self):
        """The distinguishing property of 'direct': energy without local strain."""
        rho = torch.full((16, 16), 1.3, dtype=DT)
        vx = torch.full((16, 16), 0.2, dtype=DT)
        vy = torch.zeros(16, 16, dtype=DT)
        cfg = _cfg(output_mode="direct", log_e_mean=-3.0, log_e_std=1.0)
        e = predict_e_kh(_net(), cfg, rho, vx, vy, 1 / 16, 1 / 16)
        assert e.max() > 0

    def test_unknown_mode_raises(self):
        p = _prims()
        with pytest.raises(ValueError, match="output_mode"):
            predict_e_kh(_net(), _cfg(output_mode="nope"),
                         p[..., 0], p[..., 1], p[..., 2], 1 / 16, 1 / 16)


class TestClosure:
    def test_protocol_shapes(self):
        p = _prims()
        c = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)
        tau, alpha = c(p)
        assert tau.shape == (16, 16, 4, 4)
        assert alpha.shape == (16, 16)
        assert torch.allclose(alpha, torch.ones_like(alpha))

    def test_transversality(self):
        """u_mu tau^{mu nu} = 0, i.e. tau^{0 nu} = v_i tau^{i nu}."""
        p = _prims()
        tau, _ = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)(p)
        expect = p[..., 1:2] * tau[..., 1, :] + p[..., 2:3] * tau[..., 2, :]
        assert torch.allclose(tau[..., 0, :], expect, atol=1e-14)

    def test_symmetric_spatial_block(self):
        p = _prims()
        tau, _ = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)(p)
        assert torch.allclose(tau[..., 1, 2], tau[..., 2, 1])

    def test_zz_component_is_zero(self):
        """2D SRHD has tau^zz = 0 identically (see assemble_tau)."""
        p = _prims()
        tau, _ = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)(p)
        assert torch.allclose(tau[..., 3, 3], torch.zeros_like(tau[..., 3, 3]))

    def test_energy_roundtrip(self):
        """subgrid_energy_from_stress must invert the assembled stress."""
        p = _prims()
        cfg = _cfg()
        c = LearnedMInITClosure(_net(), cfg, 1 / 16, 1 / 16)
        tau, _ = c(p)
        expect = 0.5 * (cfg.beta_xx + cfg.beta_yy) * c.e_KH
        assert torch.allclose(subgrid_energy_from_stress(tau), expect)

    def test_is_stateless(self):
        """Memoryless: no advance() hook, and repeated calls must agree."""
        c = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)
        assert not hasattr(c, "advance")
        p = _prims()
        a, _ = c(p)
        b, _ = c(p)
        assert torch.allclose(a, b)

    def test_no_grad_leak(self):
        """The solver must not accumulate a graph through the closure."""
        tau, _ = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)(_prims())
        assert not tau.requires_grad

    def test_save_load_roundtrip(self, tmp_path):
        p = _prims()
        net = _net()
        net.set_normalization(torch.randn(N_INVARIANTS, dtype=DT),
                              torch.rand(N_INVARIANTS, dtype=DT) + 0.5)
        cfg = _cfg(width=8)
        c = LearnedMInITClosure(net, cfg, 1 / 16, 1 / 16)
        before, _ = c(p)

        path = tmp_path / "coeff.pt"
        c.save(path)
        c2 = LearnedMInITClosure.load(path, 1 / 16, 1 / 16)
        after, _ = c2(p)
        assert torch.allclose(before, after), "normalisation or weights lost on reload"

    def test_matches_assemble_tau(self):
        """The closure must be exactly beta_ij * e_KH, nothing else."""
        p = _prims()
        cfg = _cfg()
        c = LearnedMInITClosure(_net(), cfg, 1 / 16, 1 / 16)
        tau, _ = c(p)
        assert torch.allclose(tau, assemble_tau(c.e_KH, p, cfg.coefficients()))


class TestSolverIntegration:
    def test_runs_in_solver(self):
        """The real gate: it must couple to RHDSolver without special-casing."""
        from src.solver import RHDSolver

        N = 32
        y = torch.linspace(-0.5, 0.5, N, dtype=DT).unsqueeze(-1)
        rho = 1.5 + 0.5 * torch.tanh((0.25 - y.abs()) / 0.05)
        vx = 0.4 * torch.tanh((0.25 - y.abs()) / 0.05)
        solver = RHDSolver(N, N, cfl=0.3, dtype=DT, device=torch.device("cpu"))
        solver.initialize(
            rho.expand(N, N).contiguous(),
            vx.expand(N, N).contiguous(),
            torch.zeros(N, N, dtype=DT),
            torch.full((N, N), 2.5, dtype=DT),
        )

        c = LearnedMInITClosure(_net(), _cfg(), solver.dx, solver.dy)
        solver.evolve(solver.t + 0.05, sgs_closure=c,
                      progress_bar=False, verbose=False)

        assert torch.isfinite(solver.prims).all()
        assert c.e_KH is not None and (c.e_KH >= 0).all()

    def test_no_advance_hook_is_harmless(self):
        """MInITClosure needs advance(); this one must not, and evolve must cope."""
        from src.solver import RHDSolver

        N = 32
        solver = RHDSolver(N, N, cfl=0.3, dtype=DT, device=torch.device("cpu"))
        solver.initialize(
            torch.ones(N, N, dtype=DT),
            torch.zeros(N, N, dtype=DT),
            torch.zeros(N, N, dtype=DT),
            torch.full((N, N), 2.5, dtype=DT),
        )
        c = LearnedMInITClosure(_net(), _cfg(), solver.dx, solver.dy)
        assert not hasattr(c, "advance")
        solver.evolve(solver.t + 0.02, sgs_closure=c,
                      progress_bar=False, verbose=False)
        assert torch.isfinite(solver.prims).all()


class TestRolloutWiring:
    """The learned runs share MInIT's run/history dicts, so they inherit its
    spectra, bar chart, panels and animation. They must not inherit its *name*,
    and they must not be dropped from the figures the way MInIT (paper) once was.
    """

    def test_shipped_checkpoints_have_a_style(self):
        from src.evaluate_rollout import _MINIT_STYLE

        for name in ("residual", "direct"):
            assert f"learned-{name}" in _MINIT_STYLE

    def test_styles_stay_visually_distinct(self):
        from src.evaluate_rollout import _MINIT_STYLE

        keys = [(v["color"], v["ls"]) for v in _MINIT_STYLE.values()]
        assert len(set(keys)) == len(keys), "two runs would draw identically"

    def test_label_and_key_distinguish_learned_from_minit(self):
        from src.evaluate_rollout import _variant_key, _variant_label

        assert _variant_label("paper") == "MInIT (paper)"
        assert _variant_key("paper") == "minit_paper"
        assert _variant_label("learned-residual") == "learned-coeff (residual)"
        assert _variant_key("learned-residual") == "learned_residual"
        assert "MInIT" not in _variant_label("learned-residual")

    def test_collect_runs_labels_learned_correctly(self):
        """The animation panel title comes from here."""
        from src.evaluate_rollout import _collect_runs

        hist = {"rho": torch.zeros(3, 8, 8)}
        runs = _collect_runs(None, None, None,
                             {"paper": hist, "learned-residual": hist})
        labels = [r[0] for r in runs]
        assert "MInIT (paper)" in labels
        assert "learned-coeff (residual)" in labels

    def test_missing_checkpoint_raises_not_silently_skipped(self):
        """A typo in the submit file must fail loudly, not produce a quiet run."""
        import inspect

        from src.evaluate_rollout import evaluate_rollout

        src = inspect.getsource(evaluate_rollout)
        assert "learned coefficients" in src and "FileNotFoundError" in src


class TestPrecisionOnAccelerators:
    """cuDNN ships no float64 convolution kernels, so a double-precision conv2d
    on CUDA dies with "GET was unable to find an engine to execute this
    computation". The solver is float64, so the network must be able to run in
    float32 while serving float64 fields. This crashed a submitted rollout that
    a CPU-only smoke test had passed.
    """

    def test_float32_net_accepts_float64_input(self):
        """The exact combination that failed: double fields, single-precision net."""
        net = CoefficientNet(width=8).to(dtype=torch.float32)
        out = net(torch.randn(12, 12, N_INVARIANTS, dtype=torch.float64))
        assert out.dtype == torch.float64, "caller's dtype must be preserved"
        assert torch.isfinite(out).all()

    def test_forward_preserves_input_dtype(self):
        for net_dt in (torch.float32, torch.float64):
            net = CoefficientNet(width=8).to(dtype=net_dt)
            for in_dt in (torch.float32, torch.float64):
                out = net(torch.randn(8, 8, N_INVARIANTS, dtype=in_dt))
                assert out.dtype == in_dt

    def test_float64_is_kept_on_every_device(self):
        """_conv2d_gemm bypasses cuDNN, so double precision survives on GPU too."""
        c = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16).to("cpu")
        assert c.net.conv1.weight.dtype == torch.float64

    def test_explicit_dtype_overrides_the_choice(self):
        c = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)
        c.to("cpu", dtype=torch.float32)
        assert c.net.conv1.weight.dtype == torch.float32

    def test_float32_net_still_produces_float64_stress(self):
        """End to end: the tensor handed to the solver must stay float64."""
        c = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)
        c.to("cpu", dtype=torch.float32)
        tau, alpha = c(_prims())
        assert tau.dtype == torch.float64
        assert alpha.dtype == torch.float64
        assert torch.isfinite(tau).all()

    def test_float32_and_float64_agree_to_single_precision(self):
        """Dropping to float32 must not change e_KH beyond ~1e-6 relative."""
        p = _prims()
        net = _net()
        c64 = LearnedMInITClosure(net, _cfg(), 1 / 16, 1 / 16).to("cpu")
        e64 = predict_e_kh(c64.net, c64.cfg, p[..., 0], p[..., 1], p[..., 2],
                           1 / 16, 1 / 16)
        c32 = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)
        c32.to("cpu", dtype=torch.float32)
        e32 = predict_e_kh(c32.net, c32.cfg, p[..., 0], p[..., 1], p[..., 2],
                           1 / 16, 1 / 16)
        e64, e32 = e64.detach(), e32.detach()
        scale = e64.abs().max().clamp(min=1e-30)
        assert float((e64 - e32).abs().max() / scale) < 1e-5

    def test_load_places_net_on_requested_device(self, tmp_path):
        path = tmp_path / "c.pt"
        LearnedMInITClosure(_net(), _cfg(width=8), 1 / 16, 1 / 16).save(path)
        c = LearnedMInITClosure.load(path, 1 / 16, 1 / 16, device="cpu")
        assert c.net.conv1.weight.device.type == "cpu"
        assert c.net.conv1.weight.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_cuda_runs_in_float64(self):
        """The real regression, when a GPU is actually reachable."""
        c = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16).to("cuda")
        assert c.net.conv1.weight.dtype == torch.float64
        tau, _ = c(_prims().cuda())
        assert tau.dtype == torch.float64
        assert torch.isfinite(tau).all()


class TestGemmConvolution:
    """``F.conv2d`` cannot run on this cluster's V100s: the installed PyTorch
    ships cuDNN 9.24, which refuses SM < 7.5, so every conv2d dies with "GET was
    unable to find an engine to execute this computation" in *both* float32 and
    float64. _conv2d_gemm routes through cuBLAS instead. It must be numerically
    interchangeable with conv2d, or the trained checkpoint stops meaning what it
    meant when it was fitted.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_matches_torch_conv2d(self, dtype):
        from src.minit_learned import _conv2d_gemm

        torch.manual_seed(0)
        x = torch.randn(2, 6, 11, 13, dtype=dtype)
        w = torch.randn(4, 6, 3, 3, dtype=dtype)
        b = torch.randn(4, dtype=dtype)
        tol = 1e-12 if dtype is torch.float64 else 1e-5
        assert torch.allclose(_conv2d_gemm(x, w, b),
                              F.conv2d(x, w, b, padding=0), atol=tol, rtol=tol)

    def test_matches_conv2d_without_bias(self):
        from src.minit_learned import _conv2d_gemm

        x = torch.randn(1, 3, 9, 9, dtype=DT)
        w = torch.randn(5, 3, 3, 3, dtype=DT)
        assert torch.allclose(_conv2d_gemm(x, w, None),
                              F.conv2d(x, w, None, padding=0), atol=1e-12)

    def test_handles_1x1_kernel(self):
        """The output head is a 1x1 convolution."""
        from src.minit_learned import _conv2d_gemm

        x = torch.randn(1, 7, 8, 8, dtype=DT)
        w = torch.randn(1, 7, 1, 1, dtype=DT)
        b = torch.randn(1, dtype=DT)
        out = _conv2d_gemm(x, w, b)
        assert out.shape == (1, 1, 8, 8)
        assert torch.allclose(out, F.conv2d(x, w, b), atol=1e-12)

    def test_network_forward_never_calls_F_conv2d(self):
        """A regression guard: reintroducing F.conv2d breaks every GPU run."""
        import inspect

        from src.minit_learned import CoefficientNet as CN

        src = inspect.getsource(CN.forward)
        assert "_conv2d(" in src
        assert "self.conv1(" not in src and "self.head(" not in src

    def test_trained_checkpoint_still_loads(self, tmp_path):
        """nn.Conv2d modules are kept as weight containers, so state_dict keys
        are unchanged and an already-fitted checkpoint stays valid."""
        net = _net()
        c = LearnedMInITClosure(net, _cfg(width=8), 1 / 16, 1 / 16)
        p = _prims()
        before, _ = c(p)
        path = tmp_path / "c.pt"
        c.save(path)
        assert "conv1.weight" in torch.load(path, weights_only=False)["state_dict"]
        after, _ = LearnedMInITClosure.load(path, 1 / 16, 1 / 16)(p)
        assert torch.allclose(before, after)


class TestConvDispatch:
    """_conv2d picks a route that runs on the current device. All three tiers
    must agree numerically, or which machine you run on changes the model.
    """

    def test_dispatch_matches_conv2d_on_cpu(self):
        from src.minit_learned import _conv2d

        x = torch.randn(1, 4, 9, 9, dtype=DT)
        w = torch.randn(3, 4, 3, 3, dtype=DT)
        b = torch.randn(3, dtype=DT)
        assert torch.allclose(_conv2d(x, w, b), F.conv2d(x, w, b), atol=1e-12)

    def test_cpu_is_reported_usable_without_touching_cudnn(self):
        from src.minit_learned import _cudnn_conv_usable

        assert _cudnn_conv_usable(torch.device("cpu")) is True

    def test_probe_result_is_cached(self):
        from src.minit_learned import _CUDNN_CONV_OK, _cudnn_conv_usable

        _CUDNN_CONV_OK.clear()
        _cudnn_conv_usable(torch.device("cpu"))
        assert ("cpu", None) in _CUDNN_CONV_OK

    def test_gemm_fallback_is_still_reachable_and_exact(self):
        """Tier 3 must stay correct even though tier 2 normally wins."""
        from src.minit_learned import _conv2d_gemm

        x = torch.randn(1, 4, 9, 9, dtype=DT)
        w = torch.randn(3, 4, 3, 3, dtype=DT)
        b = torch.randn(3, dtype=DT)
        assert torch.allclose(_conv2d_gemm(x, w, b), F.conv2d(x, w, b), atol=1e-12)


class TestHybridClosure:
    """TBNN shape x MiNNit amplitude: energy must be MiNNit's, the tensor the
    TBNN's (or MiNNit's own where the TBNN energy is too small to define one)."""

    def _hybrid(self, scale_cells_to_zero=False):
        from src.minit_learned import HybridClosure

        learned = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)
        p = _prims()
        tau_t, _ = LearnedMInITClosure(_net(seed=3), _cfg(beta_xx=1.2, beta_yy=0.8,
                                                         beta_xy=0.3), 1 / 16, 1 / 16)(p)
        tau_t = tau_t * 7.0                      # different amplitude and shape
        if scale_cells_to_zero:
            tau_t[:4] = 0.0                      # TBNN energy vanishes here

        class Stub:
            def __call__(self, prims):
                return tau_t, torch.ones(prims.shape[:-1], dtype=DT)

        # clip_mode=None: these tests isolate the rescale; TestHybridClip covers the clip
        return HybridClosure(Stub(), learned, clip_mode=None), learned, tau_t, p

    def test_energy_is_minnit_and_shape_is_tbnn(self):
        h, learned, tau_t, p = self._hybrid()
        tau, _ = h(p)
        e = 0.5 * (tau[..., 1, 1] + tau[..., 2, 2])
        ok = 0.5 * (tau_t[..., 1, 1] + tau_t[..., 2, 2]) > h.floor * learned.e_KH.mean()
        assert torch.allclose(e[ok], learned.e_KH[ok])
        ratio = tau[..., 1, 2][ok] / tau[..., 1, 1][ok]
        assert torch.allclose(ratio, (tau_t[..., 1, 2] / tau_t[..., 1, 1])[ok])

    def test_falls_back_to_minnit_where_tbnn_energy_vanishes(self):
        h, learned, _, p = self._hybrid(scale_cells_to_zero=True)
        tau, _ = h(p)
        tau_l, _ = learned(p)
        assert torch.allclose(tau[:4], tau_l[:4])
        assert h.fallback_fraction >= 4 / 16

    def test_transversality_preserved(self):
        h, _, _, p = self._hybrid(scale_cells_to_zero=True)
        tau, _ = h(p)
        expect = p[..., 1:2] * tau[..., 1, :] + p[..., 2:3] * tau[..., 2, :]
        assert torch.allclose(tau[..., 0, :], expect, atol=1e-14)


class TestHybridClip:
    """The rescale e_MiNNit/e_TBNN reached ~760x in a rollout, putting the stress up to
    170x over the realizability bound; the hybrid then diverged. The post-rescale clip
    must bound it, and clip_mode=None must keep the old behaviour for A/B runs."""

    def _closures(self):
        from src.minit_learned import HybridClosure

        learned = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)
        p = _prims()
        tau_t, _ = learned(p)
        tau_t = tau_t.clone()
        tau_t[:4] *= 1e-3                        # tiny TBNN energy -> huge rescale factor

        class Stub:
            def __call__(self, prims):
                return tau_t, torch.ones(prims.shape[:-1], dtype=DT)

        return p, (lambda **kw: HybridClosure(Stub(), learned, floor=1e-6, **kw))

    def test_clipped_stress_respects_the_bound(self):
        p, make = self._closures()
        learned_gain = make()
        learned_gain.learned.cfg.gain = 1e4      # force MiNNit energy far above the bound
        tau, _ = learned_gain(p)
        bound = 0.5 * p[..., 3]
        assert (torch.norm(tau, dim=(-2, -1)) <= bound * (1 + 1e-12)).all()

    def test_clip_off_keeps_the_old_unbounded_behaviour(self):
        p, make = self._closures()
        h = make(clip_mode=None)
        h.learned.cfg.gain = 1e4
        tau, _ = h(p)
        assert (torch.norm(tau, dim=(-2, -1)) > 0.5 * p[..., 3]).any()
        assert h.max_scale > 100


class TestHybridAblations:
    """Shape/energy ablations swap one factor of the hybrid for another closure's."""

    def test_energy_closure_sets_the_energy_and_the_fallback(self):
        from src.minit_learned import HybridClosure

        p = _prims()
        learned = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)
        tau_src, _ = LearnedMInITClosure(_net(seed=5), _cfg(), 1 / 16, 1 / 16)(p)
        tau_src = tau_src * 3.0
        shape = learned                           # any closure works as the shape

        class Src:
            def __call__(self, prims):
                return tau_src, torch.ones(prims.shape[:-1], dtype=DT)

        h = HybridClosure(shape, learned, clip_mode=None, energy_closure=Src())
        tau, _ = h(p)
        e = 0.5 * (tau[..., 1, 1] + tau[..., 2, 2])
        e_src = 0.5 * (tau_src[..., 1, 1] + tau_src[..., 2, 2])
        assert torch.allclose(e, e_src.clamp(min=0.0))

    def test_set_time_reaches_every_wrapped_closure(self):
        from src.minit_learned import HybridClosure

        seen = []

        class Timed:
            def set_time(self, t):
                seen.append(t)

        learned = LearnedMInITClosure(_net(), _cfg(), 1 / 16, 1 / 16)
        HybridClosure(Timed(), learned, energy_closure=Timed()).set_time(3.5)
        assert seen == [3.5, 3.5]
