"""§8.2 Model smoke tests for ml_model.py."""
from __future__ import annotations

import math
import tempfile
import os

import pytest
import torch

from ml_model import StrategicModel, TOTAL_FEATURES, N_FRIENDLY, NUM_ROLES, NUM_OBJECTIVES, NUM_STANCES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    torch.manual_seed(42)
    return StrategicModel()


@pytest.fixture
def dummy_input():
    return torch.randn(TOTAL_FEATURES)


@pytest.fixture
def dummy_batch():
    return torch.randn(4, TOTAL_FEATURES)


# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------

class TestOutputShapes:
    """Verify forward pass produces correct output shapes: (10,2), (10,5), (10,), (10,), (10,), (10,3), scalar."""

    def test_single_input_shapes(self, model, dummy_input):
        role, obj, tp, ap, cp, stance, value = model(dummy_input)
        assert role.shape == (N_FRIENDLY, NUM_ROLES)         # (10, 2)
        assert obj.shape == (N_FRIENDLY, NUM_OBJECTIVES)     # (10, 5)
        assert tp.shape == (N_FRIENDLY,)                     # (10,)
        assert ap.shape == (N_FRIENDLY,)                     # (10,)
        assert cp.shape == (N_FRIENDLY,)                     # (10,)
        assert stance.shape == (N_FRIENDLY, NUM_STANCES)     # (10, 3)
        assert value.shape == ()                             # scalar

    def test_batch_input_shapes(self, model, dummy_batch):
        role, obj, tp, ap, cp, stance, value = model(dummy_batch)
        B = dummy_batch.size(0)
        assert role.shape == (B, N_FRIENDLY, NUM_ROLES)
        assert obj.shape == (B, N_FRIENDLY, NUM_OBJECTIVES)
        assert tp.shape == (B, N_FRIENDLY)
        assert ap.shape == (B, N_FRIENDLY)
        assert cp.shape == (B, N_FRIENDLY)
        assert stance.shape == (B, N_FRIENDLY, NUM_STANCES)
        assert value.shape == (B,)


# ---------------------------------------------------------------------------
# Softmax sum-to-one tests
# ---------------------------------------------------------------------------

class TestSoftmaxSumsToOne:
    """Verify softmax outputs sum to 1.0 per unit (role, objective, movement stance heads)."""

    def test_role_sums(self, model, dummy_input):
        role, *_ = model(dummy_input)
        sums = role.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(N_FRIENDLY), atol=1e-5)

    def test_objective_sums(self, model, dummy_input):
        _, obj, *_ = model(dummy_input)
        sums = obj.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(N_FRIENDLY), atol=1e-5)

    def test_stance_sums(self, model, dummy_input):
        *_, stance, _value = model(dummy_input)
        sums = stance.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(N_FRIENDLY), atol=1e-5)

    def test_batch_role_sums(self, model, dummy_batch):
        role, *_ = model(dummy_batch)
        sums = role.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# Sigmoid range tests
# ---------------------------------------------------------------------------

class TestSigmoidRange:
    """Verify sigmoid outputs are in [0, 1] (combat preference head)."""

    def test_combat_pref_range(self, model, dummy_input):
        *_, cp, _stance, _value = model(dummy_input)
        assert (cp >= 0.0).all()
        assert (cp <= 1.0).all()

    def test_combat_pref_range_batch(self, model, dummy_batch):
        *_, cp, _stance, _value = model(dummy_batch)
        assert (cp >= 0.0).all()
        assert (cp <= 1.0).all()


# ---------------------------------------------------------------------------
# Target priority multiplier range tests
# ---------------------------------------------------------------------------

class TestTargetPriorityRange:
    """Verify target priority multipliers are in [exp(-3), exp(3)] ≈ [0.05, 20.1]."""

    def test_multiplier_bounds(self, model, dummy_input):
        _, _, tp, *_ = model(dummy_input)
        assert (tp >= math.exp(-3.0) - 1e-6).all()
        assert (tp <= math.exp(3.0) + 1e-6).all()

    def test_multiplier_bounds_extreme_input(self, model):
        """Even with extreme inputs, multipliers stay clamped."""
        x = torch.ones(TOTAL_FEATURES) * 100.0
        _, _, tp, *_ = model(x)
        assert (tp >= math.exp(-3.0) - 1e-6).all()
        assert (tp <= math.exp(3.0) + 1e-6).all()

    def test_multiplier_near_one_at_init(self):
        """Near-zero init weights should produce multipliers near 1.0."""
        torch.manual_seed(0)
        m = StrategicModel()
        x = torch.zeros(TOTAL_FEATURES)
        _, _, tp, *_ = m(x)
        # With zero input and small init weights, raw output ≈ bias (small),
        # so exp(clamp(small, -3, 3)) ≈ exp(small) ≈ 1.0
        assert torch.allclose(tp, torch.ones(N_FRIENDLY), atol=0.5)


# ---------------------------------------------------------------------------
# Save / load tests
# ---------------------------------------------------------------------------

class TestSaveLoad:
    """Verify the model can be saved and loaded (for checkpoint pool)."""

    def test_state_dict_roundtrip(self, model, dummy_input):
        out_before = model(dummy_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), path)

            loaded = StrategicModel()
            loaded.load_state_dict(torch.load(path, weights_only=True))
            loaded.eval()

        out_after = loaded(dummy_input)
        for before, after in zip(out_before, out_after):
            assert torch.allclose(before, after, atol=1e-6)

    def test_full_model_save_load(self, model, dummy_input):
        out_before = model(dummy_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_full.pt")
            torch.save(model, path)
            loaded = torch.load(path, weights_only=False)

        out_after = loaded(dummy_input)
        for before, after in zip(out_before, out_after):
            assert torch.allclose(before, after, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow test
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Verify gradients flow through all heads."""

    def test_all_heads_have_gradients(self, model, dummy_input):
        dummy_input.requires_grad_(False)
        role, obj, tp, ap, cp, stance, value = model(dummy_input)

        # Use log of first column to get non-trivial gradients through softmax
        # (sum of softmax is constant=1, so its gradient is zero)
        loss = (role[:, 0].log().sum()
                + obj[:, 0].log().sum()
                + tp.sum()
                + ap.sum()
                + cp.log().sum()
                + stance[:, 0].log().sum()
                + value)
        loss.backward()

        # Every parameter should have a gradient
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
