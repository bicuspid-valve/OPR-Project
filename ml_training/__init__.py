"""ML training loop: PPO with GAE, reward shaping, opponent scheduling, checkpoints.

This package splits the monolithic ml_training.py into focused modules:

- config:      TrainingConfig, data structures, device helpers
- entropy:     EntropyTargetTuner (per-head adaptive entropy)
- rewards:     Reward computation and auxiliary prediction targets
- sampling:    Action sampling (no gradient) — single and batched
- checkpoint:  CheckpointPool, model loading, opponent scheduling
- collection:  Episode collection workers and generators
- loss:        Flat replay, PPO loss, auxiliary and planning distillation losses
- gae:         Generalized Advantage Estimation
- metrics:     TrainingMetrics, army loading and generation
- loop:        Main training loop (run_training)

All public names are re-exported here so ``from ml_training import X``
continues to work unchanged.
"""

# --- config ---
from ml_training.config import (
    TrainingConfig,
    TacticalActivationRecord,
    _TacticalInferenceRequest,
    _TacticalSamplingResult,
    _OPP_TYPE_MAP,
    _get_opponent_type_idx,
    _resolve_device,
    _force_tensor_device,
)

# --- entropy ---
from ml_training.entropy import EntropyTargetTuner

# --- rewards ---
from ml_training.rewards import (
    compute_round_reward,
    terminal_reward,
    RoundSnapshot,
    _compute_survival_fracs,
    _compute_obj_control_target,
    _make_round_snapshot,
)

# --- sampling ---
from ml_training.sampling import (
    sample_tactical_actions_no_grad,
    _batched_sample_tactical_no_grad,
)

# --- checkpoint ---
from ml_training.checkpoint import (
    EMABaseline,
    get_heuristic_fraction,
    _make_model,
    load_model_state_dict,
    CheckpointPool,
)

# --- collection ---
from ml_training.collection import (
    _WORKER_COUNT,
    _MAX_SHARED_OPPONENTS,
    _init_shared_worker,
    _collect_episodes_shared_worker,
    _collect_episodes_chunked_worker,
    _run_single_episode_tactical,
    _episode_tactical_generator,
    _run_games_batched_tactical,
)

# --- loss ---
from ml_training.loss import (
    FlatReplayResult,
    replay_tactical_log_probs_flat,
    compute_loss_flat,
)

# --- gae ---
from ml_training.gae import compute_gae

# --- metrics ---
from ml_training.metrics import (
    TrainingMetrics,
    _load_hof_armies,
    _load_hof_ml_armies,
    _generate_army_pair,
)

# --- loop ---
from ml_training.loop import run_training

# Re-export TacticalModel for callers that imported it via ml_training
from ml_model_tactical import TacticalModel
