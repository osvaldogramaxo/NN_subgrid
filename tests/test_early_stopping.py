"""Tests for EarlyStopping in train_apriori.py."""

import unittest
from unittest.mock import patch
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.train_apriori import EarlyStopping, plot_loss_curves


class TestEarlyStopping(unittest.TestCase):
    def test_early_stopping_triggers_after_patience(self):
        # Patience of 3
        es = EarlyStopping(patience=3, min_delta=0.0, mode="min")

        # Epoch 1: initial best
        self.assertFalse(es.step(1.0, epoch=1))
        self.assertEqual(es.best_score, 1.0)
        self.assertEqual(es.best_epoch, 1)
        self.assertEqual(es.counter, 0)

        # Epoch 2: no improvement (counter = 1)
        self.assertFalse(es.step(1.05, epoch=2))
        self.assertEqual(es.counter, 1)

        # Epoch 3: no improvement (counter = 2)
        self.assertFalse(es.step(1.02, epoch=3))
        self.assertEqual(es.counter, 2)

        # Epoch 4: no improvement (counter = 3 >= patience -> triggers early stop)
        self.assertTrue(es.step(1.01, epoch=4))
        self.assertEqual(es.counter, 3)
        self.assertTrue(es.early_stop)
        self.assertEqual(es.best_epoch, 1)

    def test_early_stopping_counter_resets_on_improvement(self):
        es = EarlyStopping(patience=2, min_delta=0.01, mode="min")

        self.assertFalse(es.step(1.0, epoch=1))
        self.assertEqual(es.counter, 0)

        # Epoch 2: no improvement
        self.assertFalse(es.step(1.0, epoch=2))
        self.assertEqual(es.counter, 1)

        # Epoch 3: improved beyond min_delta (1.0 - 0.01 = 0.99)
        self.assertFalse(es.step(0.95, epoch=3))
        self.assertEqual(es.counter, 0)
        self.assertEqual(es.best_score, 0.95)
        self.assertEqual(es.best_epoch, 3)

        # Epoch 4: no improvement
        self.assertFalse(es.step(0.96, epoch=4))
        self.assertEqual(es.counter, 1)

        # Epoch 5: no improvement (counter = 2 >= patience -> triggers)
        self.assertTrue(es.step(0.97, epoch=5))
        self.assertEqual(es.counter, 2)
        self.assertTrue(es.early_stop)

    def test_early_stopping_disabled_when_patience_le_zero_or_none(self):
        es_zero = EarlyStopping(patience=0)
        self.assertIsNone(es_zero.patience)
        for ep in range(1, 10):
            self.assertFalse(es_zero.step(10.0, epoch=ep))

        es_neg = EarlyStopping(patience=-5)
        self.assertIsNone(es_neg.patience)
        for ep in range(1, 10):
            self.assertFalse(es_neg.step(10.0, epoch=ep))

        es_none = EarlyStopping(patience=None)
        self.assertIsNone(es_none.patience)
        for ep in range(1, 10):
            self.assertFalse(es_none.step(10.0, epoch=ep))

    def test_patience_defaults_to_2x_lr_patience(self):
        def resolve_patience(lr_patience: int, patience: int | None) -> int:
            return 2 * lr_patience if patience is None else patience

        # Default: 2x lr_patience
        self.assertEqual(resolve_patience(50, None), 100)
        self.assertEqual(resolve_patience(100, None), 200)

        # Explicit override
        self.assertEqual(resolve_patience(50, 35), 35)
        self.assertEqual(resolve_patience(100, 0), 0)

    @patch("src.train_apriori.plot_pred_vs_truth")
    @patch("src.train_apriori.plot_loss_curves")
    @patch("src.train_apriori.TBNNDataset")
    @patch("src.train_apriori.train_epoch")
    @patch("src.train_apriori.validate")
    @patch("torch.jit.trace")
    def test_train_function_early_stopping_integration(
        self, _mock_trace, mock_validate, mock_train_epoch, mock_dataset_cls,
        _mock_plot_loss, _mock_plot_pred
    ):
        import tempfile
        import torch
        from src.train_apriori import train

        from src.train_apriori import SnapshotView

        # Stand-in dataset: 4 snapshots on a 2x2 coarse grid. A real object
        # rather than a MagicMock, because train() now splits it by snapshot
        # and builds views over it.
        class _Stub:
            def __init__(self):
                n = 16
                self.invariants = torch.zeros(n, 15, dtype=torch.float64)
                self.thermo = torch.zeros(n, 2, dtype=torch.float64)
                self.basis = torch.zeros(n, 12, 4, 4, dtype=torch.float64)
                self.tau_sgs = torch.zeros(n, 4, 4, dtype=torch.float64)
                self.u_con = torch.zeros(n, 4, dtype=torch.float64)
                self.u_con[:, 0] = 1.0
                self.grad_u_cov = torch.zeros(n, 4, 4, dtype=torch.float64)
                self.rho = torch.ones(n, dtype=torch.float64)
                self.p = torch.ones(n, dtype=torch.float64)
                self.inv_mean = torch.zeros(15, dtype=torch.float64)
                self.inv_std = torch.ones(15, dtype=torch.float64)
                self.thermo_mean = torch.zeros(2, dtype=torch.float64)
                self.thermo_std = torch.ones(2, dtype=torch.float64)
                self.tau_scale = torch.tensor(1.0, dtype=torch.float64)
                self.p_scale = torch.tensor(1.0, dtype=torch.float64)
                self.snapshot_boundaries = [(i * 4, 2, 2) for i in range(4)]
                self.times = [1.0, 2.0, 3.0, 4.0]

            def __len__(self):
                return self.invariants.shape[0]

            @property
            def num_snapshots(self):
                return len(self.snapshot_boundaries)

            def snapshot_view(self, indices):
                return SnapshotView(self, indices)

        mock_dataset_cls.return_value = _Stub()

        mock_train_epoch.return_value = {
            "train_loss": 0.5, "L_tensor": 0.5, "grad_norm": 0.1, "steps": 2.0,
        }

        # Validation sequence: epoch 1 improves (1.0), epoch 2 (1.1) no improvement,
        # epoch 3 (1.2) no improvement -> with lr_patience=1, patience=2x1=2, stops at epoch 3.
        # The monitored metric is EMA-smoothed, but this sequence is monotone
        # increasing so smoothing preserves the decision.
        mock_validate.side_effect = [
            {"val_loss": 1.0, "mse": 1.0, "rel_l2": 0.1},
            {"val_loss": 1.1, "mse": 1.1, "rel_l2": 0.11},
            {"val_loss": 1.2, "mse": 1.2, "rel_l2": 0.12},
            {"val_loss": 1.3, "mse": 1.3, "rel_l2": 0.13},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            model, history = train(
                data_dir=tmpdir,
                epochs=10,
                time_range=(1.0, 4.0),
                test_time=None,
                val_every=2,
                snapshots_per_batch=2,
                lr_schedule="plateau",
                lr_patience=1,  # early stopping patience will default to 2 * 1 = 2
                patience=None,
                gamma_eos=4.0 / 3.0,
                checkpoint_dir=tmpdir,
                device_str="cpu",
                dtype_str="float64",
                hidden_dim=32,
                num_hidden_layers=1,
            )

            self.assertIsNotNone(model)
            self.assertTrue(history["early_stopped"])
            self.assertEqual(history["stopped_epoch"], 3)
            self.assertEqual(history["best_epoch"], 1)
            self.assertEqual(history["patience"], 2)
            self.assertEqual(history["lr_patience"], 1)
            train_loss_list = history["train_loss"]
            assert isinstance(train_loss_list, list)
            val_mse_list = history["val_mse"]
            assert isinstance(val_mse_list, list)
            self.assertEqual(len(train_loss_list), 3)
            self.assertEqual(len(val_mse_list), 3)

    def test_plot_loss_curves_5panel(self):
        """Test that plot_loss_curves generates a 5-panel PNG file with MSE and regime markers."""
        import tempfile
        import os

        history = {
            "train_loss": [10.0, 5.0, 2.0, 1.0, 0.5, 0.2] * 15,  # 90 epochs
            "val_loss": [12.0, 6.0, 2.5, 1.2, 0.6, 0.25] * 15,
            "val_mse": [0.1, 0.05, 0.02, 0.01, 0.005, 0.002] * 15,
            "val_rel_l2": [25.0, 15.0, 5.0, 2.0, 1.0, 0.7] * 15,
            "alpha": [1.0, 0.8, 0.5, 0.2, 0.1, 0.01] * 15,
            "scale": [1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 1e-5] * 15,
            "lr": [1e-3] * 30 + [5e-4] * 30 + [2.5e-4] * 30,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_loss.png")
            plot_loss_curves(history, out_path, include_mse=True)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 1000)

    def test_plot_loss_curves_4panel_and_short(self):
        """Test that plot_loss_curves works with 4 panels and short histories without insets."""
        import tempfile
        import os

        history = {
            "train_loss": [2.0, 1.0, 0.5],
            "val_loss": [2.5, 1.2, 0.6],
            "val_mse": [0.02, 0.01, 0.005],
            "val_rel_l2": [1.2, 0.9, 0.7],
            "alpha": [1.0, 0.9, 0.8],
            "scale": [1e-3, 5e-4, 2e-4],
            "lr": [1e-3, 1e-3, 5e-4],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "test_short.png")
            plot_loss_curves(history, out_path, include_mse=False)
            self.assertTrue(os.path.exists(out_path))
            self.assertGreater(os.path.getsize(out_path), 1000)


if __name__ == "__main__":
    unittest.main()

