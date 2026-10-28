"""CRMHD_TBNN.drop_inputs / drop_basis: dropped inputs and basis tensors cannot affect the output."""
import torch

from utils import CRMHD_TBNN


def test_dropped_inputs_and_basis_are_ignored_and_survive_checkpoint():
    torch.manual_seed(0)
    model = CRMHD_TBNN(hidden_dim=16).double()
    model.drop_inputs, model.drop_basis = (10, 11, 12, 15, 16), (9, 11)
    inv, thermo, basis = torch.randn(8, 15).double(), torch.randn(8, 2).double(), torch.randn(8, 12, 4, 4).double()
    tau, coeffs, _ = model(inv, thermo, basis)

    inv2, thermo2, basis2 = inv.clone(), thermo.clone(), basis.clone()
    inv2[:, [10, 11, 12]] += 5.0
    thermo2 += 5.0
    basis2[:, [9, 11]] += 5.0
    assert torch.allclose(model(inv2, thermo2, basis2)[0], tau)
    assert (coeffs[:, [9, 11]] == 0).all()
    inv2[:, 0] += 5.0                                           # a kept input still matters
    assert not torch.allclose(model(inv2, thermo, basis)[0], tau)

    ckpt = {"model_state_dict": model.state_dict(), "drop_inputs": [10, 11, 12, 15, 16], "drop_basis": [9, 11]}
    loaded, _ = CRMHD_TBNN.from_checkpoint(ckpt)
    assert torch.allclose(loaded(inv, thermo, basis)[0], tau)
    assert CRMHD_TBNN.from_checkpoint({"model_state_dict": model.state_dict()})[0].drop_inputs == ()


def test_shape_only_output_has_unit_trace_and_survives_checkpoint():
    from utils import minkowski_trace
    torch.manual_seed(1)
    model = CRMHD_TBNN(hidden_dim=16).double()
    model.shape_only = True
    basis = torch.randn(8, 12, 4, 4).double()
    basis[:, 0] = torch.diag(torch.tensor([0.0, 1.0, 1.0, 1.0])).double()   # h at rest, trace 3
    shape, _, _ = model(torch.randn(8, 15).double(), torch.randn(8, 2).double(), basis)
    assert torch.allclose(minkowski_trace(shape), torch.ones(8, dtype=torch.float64))
    loaded, _ = CRMHD_TBNN.from_checkpoint({"model_state_dict": model.state_dict(), "shape_only": True})
    assert loaded.shape_only
