"""§8.4 Training smoke tests for ml_training.py."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from board import Board
from models import ResolvedUnit, UnitState, Weapon
from ml_model import StrategicModel
from ml_training import (
    TrainingConfig,
    EMABaseline,
    CheckpointPool,
    RoundRecord,
    TrajectoryRound,
    compute_round_reward,
    terminal_reward,
    sample_actions_and_record,
    compute_loss,
    compute_gae,
    get_heuristic_fraction,
    run_training,
    TrainingMetrics,
)


# ---------------------------------------------------------------------------
# Fixture helpers (same pattern as test_ml_integration.py)
# ---------------------------------------------------------------------------

def _make_weapon(*, name="Gun", range_inches=24, attacks=1, ap=0,
                 deadly=0, melee=False, reliable=False, blast=0) -> Weapon:
    return Weapon(name=name, range_inches=range_inches, attacks=attacks,
                  ap=ap, deadly=deadly, melee=melee, reliable=reliable,
                  blast=blast)


def _make_resolved(*, name="TestUnit", models=5, quality=4, defense=4,
                   tough=0, points=100, weapons=None, impact=0,
                   flying=False, artillery=False, stealth=False,
                   fearless=False, fear=0, fast=False, teleport=False) -> ResolvedUnit:
    if weapons is None:
        weapons = [_make_weapon() for _ in range(models)]
    wpm = [[] for _ in range(models)]
    for i, w in enumerate(weapons):
        wpm[i % models].append(w)
    return ResolvedUnit(
        template_id="test", name=name, models=models, quality=quality,
        defense=defense, tough=tough, points=points, weapons=weapons,
        weapons_per_model=wpm, flying=flying, artillery=artillery,
        stealth=stealth, fearless=fearless, fear=fear, fast=fast,
        teleport=teleport, impact=impact,
    )


def _make_unit_state(resolved: ResolvedUnit, owner: str = "A",
                     positions: list[tuple[int, int]] | None = None,
                     ai_role: str = "killer",
                     combat_preference: str = "ranged") -> UnitState:
    us = UnitState(unit=resolved, owner=owner)
    us.ai_role = ai_role
    us.combat_preference = combat_preference
    if positions is not None:
        us.positions = list(positions)
    else:
        us.positions = [(10 + i, 5) for i in range(resolved.models)]
    return us


def _build_test_armies():
    """Build two small armies (3 units each) for testing."""
    a1 = _make_resolved(name="A_Archers", models=5, quality=4, defense=4,
                        points=100, weapons=[
                            _make_weapon(attacks=1, ap=0) for _ in range(5)
                        ])
    a2 = _make_resolved(name="A_Swords", models=3, quality=4, defense=3,
                        points=80, weapons=[
                            _make_weapon(name="Blade", melee=True,
                                         range_inches=0, attacks=2)
                            for _ in range(3)
                        ])
    a3 = _make_resolved(name="A_Lancers", models=3, quality=3, defense=3,
                        points=120, weapons=[
                            _make_weapon(attacks=2, ap=1) for _ in range(3)
                        ])

    b1 = _make_resolved(name="B_Gunners", models=5, quality=4, defense=4,
                        points=110, weapons=[
                            _make_weapon(attacks=2, ap=1) for _ in range(5)
                        ])
    b2 = _make_resolved(name="B_Tank", models=1, quality=3, defense=2,
                        points=200, tough=6, weapons=[
                            _make_weapon(attacks=6, ap=3, deadly=3),
                        ])
    b3 = _make_resolved(name="B_Scouts", models=3, quality=4, defense=5,
                        points=60, weapons=[
                            _make_weapon(attacks=1, ap=0) for _ in range(3)
                        ])
    return [a1, a2, a3], [b1, b2, b3]


# ---------------------------------------------------------------------------
# 1. Main smoke test: run N batches, verify loss doesn't explode
# ---------------------------------------------------------------------------

class TestTrainingSmokeTest:
    """§8.4: Run 10 batches of 8 games each, verify basic sanity."""

    @pytest.fixture(autouse=True)
    def setup_cleanup(self, tmp_path):
        """Use tmp_path for checkpoints to avoid polluting the project."""
        self.checkpoint_dir = str(tmp_path / "ckpts")
        yield
        # Cleanup happens automatically with tmp_path

    def test_training_runs_without_error(self):
        """10 batches x 8 games should complete without crashing."""
        torch.manual_seed(42)
        army_a, army_b = _build_test_armies()
        army_pairs = [(army_a, army_b)]

        config = TrainingConfig(
            num_batches=10,
            batch_size=8,
            lr=1e-3,
            entropy_coeff_start=0.01,
            entropy_coeff_end=0.001,
            checkpoint_interval=5,
            checkpoint_dir=self.checkpoint_dir,
        )

        model, metrics = run_training(
            config=config, army_pairs=army_pairs, verbose=False,
        )

        assert len(metrics.batch_logs) == 10

    def test_loss_does_not_explode(self):
        """Loss should remain finite across all batches."""
        torch.manual_seed(42)
        army_a, army_b = _build_test_armies()
        army_pairs = [(army_a, army_b)]

        config = TrainingConfig(
            num_batches=10,
            batch_size=8,
            checkpoint_interval=5,
            checkpoint_dir=self.checkpoint_dir,
        )

        model, metrics = run_training(
            config=config, army_pairs=army_pairs, verbose=False,
        )

        for entry in metrics.batch_logs:
            assert not (abs(entry["loss"]) > 1e6), (
                f"Loss exploded at batch {entry['batch']}: {entry['loss']}"
            )

    def test_entropy_starts_high(self):
        """Entropy should start near maximum and not immediately collapse."""
        torch.manual_seed(42)
        army_a, army_b = _build_test_armies()
        army_pairs = [(army_a, army_b)]

        config = TrainingConfig(
            num_batches=10,
            batch_size=8,
            checkpoint_interval=50,
            checkpoint_dir=self.checkpoint_dir,
        )

        model, metrics = run_training(
            config=config, army_pairs=army_pairs, verbose=False,
        )

        # First batch entropy should be well above zero
        first_entropy = metrics.batch_logs[0]["mean_entropy"]
        assert first_entropy > 0.1, (
            f"Initial entropy too low: {first_entropy:.4f}"
        )

        # Should not collapse to near-zero by batch 10
        last_entropy = metrics.batch_logs[-1]["mean_entropy"]
        assert last_entropy > 0.01, (
            f"Entropy collapsed by batch 10: {last_entropy:.4f}"
        )

    def test_win_rate_tracking(self):
        """Win rate tracking should produce values in [0, 1]."""
        torch.manual_seed(42)
        army_a, army_b = _build_test_armies()
        army_pairs = [(army_a, army_b)]

        config = TrainingConfig(
            num_batches=5,
            batch_size=8,
            checkpoint_interval=50,
            checkpoint_dir=self.checkpoint_dir,
        )

        model, metrics = run_training(
            config=config, army_pairs=army_pairs, verbose=False,
        )

        for entry in metrics.batch_logs:
            assert 0.0 <= entry["heuristic_win_rate"] <= 1.0
            assert 0.0 <= entry["selfplay_win_rate"] <= 1.0


# ---------------------------------------------------------------------------
# 2. Reward computation
# ---------------------------------------------------------------------------

class TestRewardComputation:
    def test_terminal_reward(self):
        assert terminal_reward("A", "A") == 1.0
        assert terminal_reward("B", "A") == -1.0
        assert terminal_reward("draw", "A") == 0.0
        assert terminal_reward("A", "B") == -1.0
        assert terminal_reward("B", "B") == 1.0

    def test_round_reward_neutral(self):
        """No objectives, no kills → zero reward."""
        a1 = _make_resolved(points=100)
        b1 = _make_resolved(points=100)
        us_a = _make_unit_state(a1, owner="A")
        us_b = _make_unit_state(b1, owner="B")
        board = Board()

        reward, fk, ek = compute_round_reward(
            [us_a], [us_b], board, "A", 100, 0.0, 0.0,
        )
        assert reward == 0.0

    def test_round_reward_positive_for_objective(self):
        """Controlling an objective should yield positive reward."""
        a1 = _make_resolved(points=100)
        b1 = _make_resolved(points=100)
        us_a = _make_unit_state(a1, owner="A")
        us_b = _make_unit_state(b1, owner="B")
        board = Board()
        board.objective_control[0] = "A"  # Control centre

        reward, fk, ek = compute_round_reward(
            [us_a], [us_b], board, "A", 100, 0.0, 0.0,
        )
        assert reward > 0

    def test_round_reward_positive_for_kill(self):
        """Killing an enemy unit should yield positive reward."""
        a1 = _make_resolved(points=100)
        b1 = _make_resolved(points=100)
        us_a = _make_unit_state(a1, owner="A")
        us_b = _make_unit_state(b1, owner="B")
        us_b.models_alive = 0  # enemy destroyed
        board = Board()

        reward, fk, ek = compute_round_reward(
            [us_a], [us_b], board, "A", 100, 0.0, 0.0,
        )
        assert reward > 0  # killed enemy points > 0


# ---------------------------------------------------------------------------
# 3. EMA Baseline
# ---------------------------------------------------------------------------

class TestEMABaseline:
    def test_initial_value(self):
        b = EMABaseline(alpha=0.01)
        b.update(1.0)
        assert b.get() == 1.0  # first update sets value directly

    def test_moving_average(self):
        b = EMABaseline(alpha=0.5)
        b.update(1.0)   # value = 1.0
        b.update(0.0)   # value = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert abs(b.get() - 0.5) < 1e-6

    def test_converges_toward_constant(self):
        b = EMABaseline(alpha=0.1)
        for _ in range(1000):
            b.update(5.0)
        assert abs(b.get() - 5.0) < 0.01


# ---------------------------------------------------------------------------
# 4. Opponent scheduling
# ---------------------------------------------------------------------------

class TestOpponentScheduling:
    def test_low_win_rate(self):
        assert get_heuristic_fraction(0.40) == 0.50

    def test_medium_win_rate(self):
        assert get_heuristic_fraction(0.60) == 0.30

    def test_high_win_rate(self):
        assert get_heuristic_fraction(0.70) == 0.10

    def test_boundary_55(self):
        assert get_heuristic_fraction(0.55) == 0.30

    def test_boundary_65(self):
        assert get_heuristic_fraction(0.65) == 0.30


# ---------------------------------------------------------------------------
# 5. Checkpoint pool
# ---------------------------------------------------------------------------

class TestCheckpointPool:
    def test_save_and_load(self, tmp_path):
        pool = CheckpointPool(max_size=5, save_dir=str(tmp_path / "ckpts"))
        model = StrategicModel()
        torch.manual_seed(42)

        pool.save(model, 0)
        assert len(pool) == 1

        opponent = pool.sample_opponent()
        assert opponent is not None

        # Loaded model should produce same outputs
        from ml_features import TOTAL_FEATURES
        x = torch.randn(TOTAL_FEATURES)
        with torch.no_grad():
            orig_out = model(x)
            loaded_out = opponent(x)
        for o, l in zip(orig_out, loaded_out):
            assert torch.allclose(o, l, atol=1e-6)

    def test_eviction(self, tmp_path):
        pool = CheckpointPool(max_size=3, save_dir=str(tmp_path / "ckpts"))
        model = StrategicModel()

        for i in range(5):
            pool.save(model, i)

        assert len(pool) == 3
        # Oldest two should have been evicted
        assert not (tmp_path / "ckpts" / "checkpoint_batch_000000.pt").exists()
        assert not (tmp_path / "ckpts" / "checkpoint_batch_000001.pt").exists()

    def test_empty_pool_returns_none(self, tmp_path):
        pool = CheckpointPool(max_size=5, save_dir=str(tmp_path / "ckpts"))
        assert pool.sample_opponent() is None


# ---------------------------------------------------------------------------
# 6. Action sampling produces valid log-probs
# ---------------------------------------------------------------------------

class TestActionSampling:
    def test_sampling_produces_record(self):
        """sample_actions_and_record should return a RoundRecord with finite values."""
        torch.manual_seed(42)
        model = StrategicModel()

        army_a, army_b = _build_test_armies()
        units_a = [_make_unit_state(r, owner="A",
                                    positions=[(10 + j, 5) for j in range(r.models)])
                   for r in army_a]

        from ml_features import TOTAL_FEATURES
        x = torch.randn(TOTAL_FEATURES)
        role_probs, obj_probs, target_priority, act_scores, combat_prefs, stance_probs, _value = model(x)

        mults, record = sample_actions_and_record(
            role_probs, obj_probs, target_priority, act_scores,
            combat_prefs, stance_probs, units_a, "A",
        )

        assert record.log_prob.isfinite()
        assert record.entropy.isfinite()
        assert record.entropy.item() > 0  # should have some entropy
        assert len(mults) == 10


# ---------------------------------------------------------------------------
# 7. Compute loss
# ---------------------------------------------------------------------------

class TestComputeLoss:
    def test_loss_is_finite(self):
        """Loss computation should produce a finite scalar."""
        torch.manual_seed(42)
        record = RoundRecord(
            log_prob=torch.tensor(-1.5, requires_grad=True),
            entropy=torch.tensor(0.5),
            reward=0.3,
            value=torch.tensor(0.1, requires_grad=True),
        )
        traj_round = TrajectoryRound(
            state_vec=[0.0] * 591,
            unit_actions=[(0, 0, 0, 0)] * 10,
            unit_alive_mask=[False] * 10,
            reward=0.3,
            old_log_prob=-1.5,
            old_value=0.0,
        )
        episodes = [([record], "heuristic")]
        all_trajs = [[traj_round]]
        all_advantages, all_returns = compute_gae(all_trajs)

        loss, metrics = compute_loss(
            episodes, all_trajs, all_advantages, all_returns,
            clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01,
        )
        assert loss.isfinite()
        assert "loss" in metrics
        assert "mean_entropy" in metrics
        assert "mean_reward" in metrics
        assert "value_loss" in metrics

    def test_loss_requires_grad(self):
        """Loss should be differentiable."""
        record = RoundRecord(
            log_prob=torch.tensor(-1.0, requires_grad=True),
            entropy=torch.tensor(0.5),
            reward=1.0,
            value=torch.tensor(0.2, requires_grad=True),
        )
        traj_round = TrajectoryRound(
            state_vec=[0.0] * 591,
            unit_actions=[(0, 0, 0, 0)] * 10,
            unit_alive_mask=[False] * 10,
            reward=1.0,
            old_log_prob=-1.0,
            old_value=0.0,
        )
        episodes = [([record], "heuristic")]
        all_trajs = [[traj_round]]
        all_advantages, all_returns = compute_gae(all_trajs)

        loss, _ = compute_loss(
            episodes, all_trajs, all_advantages, all_returns,
            clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01,
        )
        loss.backward()
        assert record.log_prob.grad is not None


# ---------------------------------------------------------------------------
# 8. Metrics tracking
# ---------------------------------------------------------------------------

class TestMetricsTracking:
    def test_win_rate_updates(self):
        m = TrainingMetrics()
        for _ in range(10):
            m.record_game("A", "heuristic")  # 10 wins
        for _ in range(10):
            m.record_game("B", "heuristic")  # 10 losses
        assert abs(m.heuristic_win_rate - 0.5) < 1e-6

    def test_selfplay_win_rate(self):
        m = TrainingMetrics()
        m.record_game("A", "selfplay")
        m.record_game("draw", "selfplay")
        assert abs(m.selfplay_win_rate - 0.75) < 1e-6  # 1.0 + 0.5 / 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
