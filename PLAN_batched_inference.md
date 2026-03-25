# Plan: Cross-Game Batched ML Inference for Evolution

## Problem

During `run_list_evolution(use_ml=True, ml_batch_tactical=False)`, the per-activation tactical path calls `apply_tactical_model()` individually for every unit activation across every game. With ~100 games per Swiss round, each game having ~40-80 ML calls, that's ~4000-8000 individual `model(1, 771)` forward passes per Swiss round. The model is tiny (771->256->128 trunk), so PyTorch per-call overhead (~0.25ms on Windows) dominates actual compute.

## Benchmark Results (already collected)

From `benchmark_batching.py` run on the target machine (16 logical cores, Windows 11):

**Forward pass batching speedup** (Test 1):
- batch=1: 0.243ms/sample, batch=25: 0.014ms/sample (17x), batch=50: 0.009ms/sample (27x)

**Current per-activation ML at different worker counts** (Test 3):
- 1w: 2.696s, 2w: 1.560s, 4w: 1.017s, 8w: 1.074s

**Hybrid batched (generator+coordinator per worker)** (Test 4):
- 1w: 1.752s, 2w: 1.069s, 4w: 0.792s, 8w: 0.938s

**Best hybrid (4 workers) is 22% faster than best current (4 workers)**: 0.792s vs 1.017s per Swiss round. Over 9 Swiss rounds per generation and 200 generations, this saves ~40 minutes of wall time.

**Timing breakdown** (from evolution benchmark instrumentation):
- ML forward: 33.8%, ML encode: 17.0%, Game logic: 49.2%
- 3777 forward calls per Swiss round, avg 0.388ms per forward, avg 0.194ms per encode

## Approach: Hybrid Multi-Process Generator+Coordinator

Keep ProcessPoolExecutor with 4 workers. Each worker receives a **chunk** of matchups (not individual matchups). Inside each worker, all games run as **Python generators** that yield at ML decision points. A per-worker **coordinator loop** collects all pending inference requests, batches them into a single `model(N, 771)` forward pass, and distributes results back via `gen.send()`.

```
evaluate_population (per Swiss round)
  -> split 50 matchups into 4 chunks of ~12-13 matchups
  -> pool.map(_play_matchup_batched, chunks)
       Each worker:
         1. Create ~25 game generators (coroutine mode)
         2. Prime all generators (next() -> first yield)
         3. Coordinator loop:
              while any game active:
                batch_vecs = torch.stack(all pending state_vecs)      # ~25 x 771
                batch_masks = torch.stack(all pending alive_masks)    # ~25 x 10
                outputs = model(batch_vecs, batch_masks)              # single forward
                for each game: gen.send(InferenceResult from outputs)
                  -> yields next request or raises StopIteration
         4. Return aggregated (a_wins, b_wins) per matchup
```

## What Already Exists (partially implemented)

The following work has already been done in the current codebase:

### `ml_integration_tactical.py`
- `InferenceRequest` dataclass: `state_vec` (771,), `alive_mask` (10,), `player` str
- `InferenceResult` dataclass: 7 tensors (unit_logits, role_probs, obj_probs, target_priority, combat_pref, stance_probs, value)
- `decode_tactical_result(result, friendly_units, player) -> (unit, mults)`: argmax decode from InferenceResult, same logic as apply_tactical_model lines 104-128
- Timing instrumentation: `reset_timing()`, `get_timing()`, accumulators in `apply_tactical_model`

### `game.py`
- `simulate_game()` is now a dispatcher: routes to `_simulate_game_impl()` (normal) or `_simulate_game_coroutine()` (generator) based on `ml_coroutine_mode` flag
- `_simulate_game_impl()`: the original game loop, unchanged. Also has unused `_tactical_inference_fn` parameter (can be removed or used)
- `_simulate_game_coroutine()`: a **complete generator** implementation of the game loop. At each per-activation tactical decision point, it:
  1. Builds alive mask
  2. Calls `encode_state_tactical()` to get state_vec
  3. `_ir = yield InferenceRequest(_vec, _mask, my_player)`
  4. `active, my_mults = decode_tactical_result(_ir, my_units, my_player)`
  5. Continues with normal activation logic (movement, combat, etc.)
- The coroutine variant supports non-tactical sides (strategic model, heuristic AI) without yielding

### `benchmark_batching.py`
- Contains a working coordinator implementation in `_run_games_batched()` that:
  - Creates game generators from matchup pairs
  - Primes them with `next(gen)`
  - Runs coordinator loop: stack tensors -> batched forward -> distribute via send()
  - Handles games finishing at different times (StopIteration removes from pending list)
- This can serve as the reference implementation for the production coordinator

### `evolution.py`
- `_play_matchup()` has benchmark timing instrumentation (bench flag, rest args)
- `evaluate_population()` has `bench` parameter, prints timing after Swiss round 1

## What Remains To Be Done

### 1. Create `_play_matchup_batched()` worker function in `evolution.py`

This replaces `_play_matchup` for the batched path. It receives a **list of matchup tuples** (not a single matchup) and runs all games via the generator+coordinator pattern.

```python
def _play_matchup_batched(args):
    """Worker: run a chunk of matchups using batched cross-game inference."""
    matchup_list, mode = args
    # matchup_list: [(army_i, army_j, res_i, res_j), ...]

    from game import simulate_game
    from ml_integration_tactical import InferenceResult

    games_per = GAMES_PER_MATCHUP  # 2

    # Create all game generators
    generators = []
    game_to_matchup = []  # maps generator index -> matchup index
    for m_idx, (army_i, army_j, res_i, res_j) in enumerate(matchup_list):
        for _ in range(games_per):
            sa = _make_unit_states(army_i, res_i, "A")
            sb = _make_unit_states(army_j, res_j, "B")
            gen = simulate_game(res_i, res_j, mode=mode, states_a=sa, states_b=sb,
                                ml_model_a=_g_evo_ml_model, ml_model_b=_g_evo_ml_model,
                                ml_batch_tactical=False, ml_coroutine_mode=True)
            generators.append(gen)
            game_to_matchup.append(m_idx)

    # Prime all generators
    pending = []
    results = [None] * len(generators)
    for i, gen in enumerate(generators):
        try:
            req = next(gen)
            pending.append((i, req))
        except StopIteration as e:
            results[i] = e.value

    # Coordinator loop
    import torch
    with torch.no_grad():
        while pending:
            batch_vecs = torch.stack([req.state_vec for _, req in pending])
            batch_masks = torch.stack([req.alive_mask for _, req in pending])

            (all_logits, all_role, all_obj, all_target,
             all_combat, all_stance, all_value) = _g_evo_ml_model(batch_vecs, batch_masks)

            next_pending = []
            for k, (i, req) in enumerate(pending):
                ir = InferenceResult(
                    unit_logits=all_logits[k],
                    role_probs=all_role[k],
                    obj_probs=all_obj[k],
                    target_priority=all_target[k],
                    combat_pref=all_combat[k],
                    stance_probs=all_stance[k],
                    value=all_value[k],
                )
                try:
                    next_req = generators[i].send(ir)
                    next_pending.append((i, next_req))
                except StopIteration as e:
                    results[i] = e.value
            pending = next_pending

    # Aggregate into per-matchup (a_wins, b_wins)
    matchup_results = [[0.0, 0.0] for _ in matchup_list]
    for i, result in enumerate(results):
        m_idx = game_to_matchup[i]
        if result == 'A':
            matchup_results[m_idx][0] += 1
        elif result == 'B':
            matchup_results[m_idx][1] += 1
        else:
            matchup_results[m_idx][0] += 0.5
            matchup_results[m_idx][1] += 0.5

    return matchup_results
```

### 2. Modify `evaluate_population()` in `evolution.py`

Add a new parameter `ml_coroutine_batch=False`. When True:
- Instead of building individual work items per matchup, chunk all matchups into `_WORKER_COUNT` groups
- Use `pool.map(_play_matchup_batched, chunks)` instead of `pool.map(_play_matchup, work)`
- Unpack the per-chunk results back into per-matchup results

```python
def evaluate_population(population, mode="objectives", pool=None,
                        use_ml=False, ml_batch_tactical=True,
                        bench=False, ml_coroutine_batch=False):
    ...
    for rnd in range(SWISS_ROUNDS):
        ...  # pairing logic unchanged

        if ml_coroutine_batch:
            # Chunk matchups across workers
            n_workers = _WORKER_COUNT
            chunks = [[] for _ in range(n_workers)]
            pair_chunk_map = []  # (chunk_idx, position_in_chunk)
            for p_idx, (i, j) in enumerate(round_pairs):
                c_idx = p_idx % n_workers
                chunks[c_idx].append((population[i], population[j],
                                      resolved[i], resolved[j]))
                pair_chunk_map.append((c_idx, len(chunks[c_idx]) - 1))

            chunk_work = [(chunk, mode) for chunk in chunks if chunk]

            if pool is not None:
                chunk_results = list(pool.map(_play_matchup_batched, chunk_work))
            else:
                with ProcessPoolExecutor(max_workers=n_workers) as _pool:
                    chunk_results = list(_pool.map(_play_matchup_batched, chunk_work))

            # Unpack results
            for p_idx, (i, j) in enumerate(round_pairs):
                c_idx, pos = pair_chunk_map[p_idx]
                a_wins, b_wins = chunk_results[c_idx][pos]
                population[i].wins += a_wins
                ...
        else:
            # Existing individual matchup path (unchanged)
            ...
```

### 3. Wire through `main.py`

In `run_list_evolution()`, when `use_ml=True` and `ml_batch_tactical=False`, pass `ml_coroutine_batch=True` to `evaluate_population()`.

```python
evaluate_population(population, mode=mode, pool=pool, use_ml=use_ml,
                    ml_batch_tactical=ml_batch_tactical,
                    bench=(gen == 1),
                    ml_coroutine_batch=(use_ml and not ml_batch_tactical))
```

### 4. Clean up game logic duplication (optional, lower priority)

`_simulate_game_coroutine` duplicates ~200 lines of game logic from `_simulate_game_impl`. Two options to reduce this:

**Option A (recommended)**: Keep the duplication but add a comment at the top of `_simulate_game_coroutine` noting it mirrors `_simulate_game_impl` and both must be kept in sync. This is the simplest and has no runtime cost.

**Option B**: Refactor into a shared `_simulate_game_core()` that accepts a callable for the tactical inference step. The coroutine passes a function that stores the request and returns a sentinel, then the outer generator handles the yield. This is more complex and harder to follow.

## Files to Modify

| File | Change |
|------|--------|
| `evolution.py` | Add `_play_matchup_batched()`, modify `evaluate_population()` to add `ml_coroutine_batch` path |
| `main.py` | Pass `ml_coroutine_batch=True` when `use_ml and not ml_batch_tactical` |

Already done (no further changes needed):
| File | Status |
|------|--------|
| `ml_integration_tactical.py` | InferenceRequest, InferenceResult, decode_tactical_result all exist |
| `game.py` | simulate_game dispatcher, _simulate_game_coroutine generator all exist |
| `benchmark_batching.py` | Reference coordinator implementation exists |

## Verification

1. **Correctness**: Run `run_list_evolution(use_ml=True, ml_batch_tactical=False)` for 5 generations. Compare win rates and army compositions against a run without `ml_coroutine_batch`. Results won't be identical (different random seeds from chunking) but should be statistically similar.

2. **Performance**: The `bench=True` timing instrumentation already prints after Swiss round 1. Compare wall time per generation with and without the batched path. Expected: ~22% faster per Swiss round based on benchmark results.

3. **Smoke test**: Run a single game in coroutine mode and verify it returns a valid result:
```python
gen = simulate_game(..., ml_coroutine_mode=True)
req = next(gen)
while True:
    outputs = model(req.state_vec, req.alive_mask)
    try:
        req = gen.send(InferenceResult(*outputs))
    except StopIteration as e:
        print(e.value)  # "A", "B", or "draw"
        break
```

## Key Technical Details

- **Generator protocol**: `simulate_game(..., ml_coroutine_mode=True)` returns a generator (not a string). The generator yields `InferenceRequest` at each ML decision point. The caller sends back `InferenceResult` via `gen.send()`. When the game finishes, `StopIteration.value` contains the result string.

- **Games finish at different times**: The coordinator handles this naturally — when `gen.send()` raises `StopIteration`, that game is removed from the pending list. Batch size shrinks over time.

- **Both sides use the same model**: In evolution, `ml_model_a == ml_model_b == _g_evo_ml_model`. Every activation (both A and B sides) yields an InferenceRequest. A game with 10 units per side has ~20 yields per round, ~80 per game.

- **Non-tactical sides**: If one side uses strategic model or heuristic AI, the coroutine handles it internally without yielding (only tactical per-activation decisions yield).

- **ProcessPoolExecutor worker init**: Workers still use `_init_evo_ml_worker()` to load the model. The model is used both for the batched forward pass in the coordinator and for any strategic-model round-start passes inside the coroutine.

- **`_WORKER_COUNT`**: Currently `max(1, os.cpu_count() // 2)`. Benchmarks show 4 workers is optimal on the 16-core machine (8 workers gives no benefit due to hyperthreading). Consider hardcoding or tuning.
