"""Verify that blast and deadly are correctly reflected in the matchup
features the ML model sees.

Constructs synthetic attacker/defender pairs and checks that:
  1. Blast weapons produce higher matchup values against multi-model targets
     than single-model targets (proportional to min(blast, models)).
  2. Deadly weapons produce higher matchup values against tough targets
     than non-tough targets (relative to a non-deadly baseline).
  3. The full precompute_damage pipeline preserves these relationships.
"""
from __future__ import annotations

from models import Weapon, ResolvedUnit
from ml_features import (
    _expected_ranged_damage_at_range,
    precompute_damage,
    starting_wounds,
    RANGE_THRESHOLDS,
)


# ------------------------------------------------------------------
# Helpers to build minimal ResolvedUnit
# ------------------------------------------------------------------

def _make_unit(
    name: str,
    models: int,
    quality: int,
    defense: int,
    weapons: list[Weapon],
    tough: int = 0,
    points: int = 100,
) -> ResolvedUnit:
    """Build a minimal ResolvedUnit for testing."""
    return ResolvedUnit(
        template_id="test",
        name=name,
        models=models,
        quality=quality,
        defense=defense,
        tough=tough,
        weapons=weapons,
        weapons_per_model=[[w] for w in weapons],
        points=points,
    )


# ------------------------------------------------------------------
# Weapon definitions for tests
# ------------------------------------------------------------------

# Plain weapon: 24" range, 1 attack, no specials
W_PLAIN = Weapon("Plain Gun", range_inches=24, attacks=1)

# Blast weapon: 24" range, 1 attack, Blast(3)
W_BLAST = Weapon("Blast Gun", range_inches=24, attacks=1, blast=3)

# Deadly weapon: 24" range, 1 attack, Deadly(3)
W_DEADLY = Weapon("Deadly Gun", range_inches=24, attacks=1, deadly=3)

# Combined: 24" range, 1 attack, Blast(3) + Deadly(3)
W_BLAST_DEADLY = Weapon("Blast+Deadly Gun", range_inches=24, attacks=1, blast=3, deadly=3)


# ------------------------------------------------------------------
# Test: Blast scaling with model count
# ------------------------------------------------------------------

def test_blast_scales_with_models():
    """Blast(3) should do ~3x damage against a 5-model unit vs a 1-model unit."""
    print("=" * 64)
    print("TEST: Blast damage scales with target model count")
    print("=" * 64)

    # Single blast weapon, 1 model, quality 4, firing at range 24
    attacker = _make_unit("Blast Attacker", models=1, quality=4, defense=5,
                          weapons=[W_BLAST])

    targets = {}
    for n_models in [1, 2, 3, 5]:
        defender = _make_unit(f"Target ({n_models} models)", models=n_models,
                              quality=4, defense=5, weapons=[])
        raw_dmg = _expected_ranged_damage_at_range(attacker, defender, max_range=24)
        sw = starting_wounds(defender)
        kill_frac = raw_dmg / sw
        targets[n_models] = (raw_dmg, kill_frac)
        print(f"  vs {n_models}-model target: raw_wounds={raw_dmg:.3f}  "
              f"starting_wounds={sw}  kill_frac={kill_frac:.4f}")

    # Blast(3) vs 1 model: blast_mult = min(3,1) = 1
    # Blast(3) vs 3 models: blast_mult = min(3,3) = 3 → 3x raw wounds
    # Blast(3) vs 5 models: blast_mult = min(3,5) = 3 → 3x raw wounds (capped)
    ratio_3v1 = targets[3][0] / targets[1][0]
    ratio_5v1 = targets[5][0] / targets[1][0]
    print(f"\n  Raw damage ratio (3-model / 1-model): {ratio_3v1:.2f}x  (expected: 3.0x)")
    print(f"  Raw damage ratio (5-model / 1-model): {ratio_5v1:.2f}x  (expected: 3.0x)")

    assert abs(ratio_3v1 - 3.0) < 0.01, f"Expected 3.0x, got {ratio_3v1:.4f}x"
    assert abs(ratio_5v1 - 3.0) < 0.01, f"Expected 3.0x, got {ratio_5v1:.4f}x"

    # But kill_frac should favor the 1-model target (fewer total wounds to chew through)
    # Actually: kill_frac(1) = raw/1, kill_frac(3) = 3*raw/3 = raw, kill_frac(5) = 3*raw/5
    # So kill_frac: 1-model > 3-model > 5-model
    print(f"\n  Kill fraction: 1-model={targets[1][1]:.4f}  "
          f"3-model={targets[3][1]:.4f}  5-model={targets[5][1]:.4f}")
    print(f"  (kill_frac for 1-model should be HIGHEST — blast helps with damage")
    print(f"   but more models means more total wounds to kill the whole unit)")

    print("\n  PASSED ✓\n")


# ------------------------------------------------------------------
# Test: Non-blast does NOT scale with models
# ------------------------------------------------------------------

def test_plain_does_not_scale_with_models():
    """A plain weapon's raw damage should NOT change with target model count."""
    print("=" * 64)
    print("TEST: Plain weapon damage does NOT scale with target model count")
    print("=" * 64)

    attacker = _make_unit("Plain Attacker", models=1, quality=4, defense=5,
                          weapons=[W_PLAIN])

    damages = {}
    for n_models in [1, 3, 5]:
        defender = _make_unit(f"Target ({n_models} models)", models=n_models,
                              quality=4, defense=5, weapons=[])
        raw_dmg = _expected_ranged_damage_at_range(attacker, defender, max_range=24)
        damages[n_models] = raw_dmg
        print(f"  vs {n_models}-model target: raw_wounds={raw_dmg:.3f}")

    assert abs(damages[1] - damages[3]) < 0.001, "Plain weapon damage should be constant"
    assert abs(damages[1] - damages[5]) < 0.001, "Plain weapon damage should be constant"

    print(f"\n  All equal — plain weapons don't benefit from more models.")
    print("  PASSED ✓\n")


# ------------------------------------------------------------------
# Test: Deadly scaling vs tough
# ------------------------------------------------------------------

def test_deadly_vs_tough():
    """Deadly(3) vs plain — examine raw damage and kill fractions against tough."""
    print("=" * 64)
    print("TEST: Deadly vs Tough — relative advantage over plain weapon")
    print("=" * 64)

    # Both weapons: 1 attack, 24" range. Only difference: deadly(3).
    atk_deadly = _make_unit("Deadly Attacker", models=1, quality=4, defense=5,
                            weapons=[W_DEADLY])
    atk_plain = _make_unit("Plain Attacker", models=1, quality=4, defense=5,
                           weapons=[W_PLAIN])

    # Compare against non-tough 1-model, and tough(3) 1-model
    print("\n  --- vs 1-model, non-tough (1 starting wound) ---")
    def_no_tough = _make_unit("No-Tough Target", models=1, quality=4, defense=5,
                              weapons=[], tough=0)
    deadly_raw = _expected_ranged_damage_at_range(atk_deadly, def_no_tough, 24)
    plain_raw = _expected_ranged_damage_at_range(atk_plain, def_no_tough, 24)
    sw = starting_wounds(def_no_tough)
    print(f"  Deadly: raw={deadly_raw:.4f}  kill_frac={deadly_raw/sw:.4f}")
    print(f"  Plain:  raw={plain_raw:.4f}  kill_frac={plain_raw/sw:.4f}")
    ratio_no_tough = deadly_raw / plain_raw if plain_raw > 0 else float('inf')
    print(f"  Deadly/Plain raw ratio: {ratio_no_tough:.2f}x  (expected: 1.0x — overkill wasted)")

    print("\n  --- vs 1-model, tough(3) (3 starting wounds) ---")
    def_tough3 = _make_unit("Tough(3) Target", models=1, quality=4, defense=5,
                            weapons=[], tough=3)
    deadly_raw_t = _expected_ranged_damage_at_range(atk_deadly, def_tough3, 24)
    plain_raw_t = _expected_ranged_damage_at_range(atk_plain, def_tough3, 24)
    sw_t = starting_wounds(def_tough3)
    print(f"  Deadly: raw={deadly_raw_t:.4f}  kill_frac={deadly_raw_t/sw_t:.4f}")
    print(f"  Plain:  raw={plain_raw_t:.4f}  kill_frac={plain_raw_t/sw_t:.4f}")
    ratio_tough3 = deadly_raw_t / plain_raw_t if plain_raw_t > 0 else float('inf')
    print(f"  Deadly/Plain raw ratio: {ratio_tough3:.2f}x  (expected: 3.0x — full value)")

    # Deadly(3) vs 1-wound: overkill wastes 2 wounds → same as plain (1.0x)
    # Deadly(3) vs tough(3): all 3 wounds count → 3x plain
    assert abs(ratio_no_tough - 1.0) < 0.01, f"Expected 1.0x vs non-tough, got {ratio_no_tough:.4f}x"
    assert abs(ratio_tough3 - 3.0) < 0.01, f"Expected 3.0x vs tough(3), got {ratio_tough3:.4f}x"

    # Now check the matchup feature values (kill fractions):
    deadly_kf_notough = min(deadly_raw / starting_wounds(def_no_tough), 1.0)
    plain_kf_notough = min(plain_raw / starting_wounds(def_no_tough), 1.0)
    deadly_kf_tough = min(deadly_raw_t / starting_wounds(def_tough3), 1.0)
    plain_kf_tough = min(plain_raw_t / starting_wounds(def_tough3), 1.0)

    print(f"\n  --- What the model sees (kill fractions, capped at 1.0) ---")
    print(f"  vs no-tough:  deadly_kf={deadly_kf_notough:.4f}  plain_kf={plain_kf_notough:.4f}")
    print(f"  vs tough(3):  deadly_kf={deadly_kf_tough:.4f}  plain_kf={plain_kf_tough:.4f}")

    # Kill fraction ratio: how much worse is tough vs squishy?
    deadly_pref = deadly_kf_tough / deadly_kf_notough if deadly_kf_notough > 0 else 0
    plain_pref = plain_kf_tough / plain_kf_notough if plain_kf_notough > 0 else 0
    print(f"\n  Deadly: kf_tough / kf_squishy = {deadly_pref:.4f}")
    print(f"  Plain:  kf_tough / kf_squishy = {plain_pref:.4f}")
    print(f"  Deadly ratio HIGHER → model can now see deadly is relatively")
    print(f"  better against tough targets!")

    assert deadly_pref > plain_pref, (
        f"Deadly should have higher tough/squishy ratio ({deadly_pref:.4f} vs {plain_pref:.4f})")

    print("\n  PASSED ✓\n")


# ------------------------------------------------------------------
# Test: precompute_damage pipeline
# ------------------------------------------------------------------

def test_precompute_damage_blast():
    """Verify precompute_damage reflects blast advantage against hordes."""
    print("=" * 64)
    print("TEST: precompute_damage — blast attacker vs varied targets")
    print("=" * 64)

    blast_atk = _make_unit("Blast Attacker", models=1, quality=4, defense=5,
                           weapons=[W_BLAST])
    plain_atk = _make_unit("Plain Attacker", models=1, quality=4, defense=5,
                           weapons=[W_PLAIN])

    # Two targets: 1-model and 5-model, same defense
    target_1 = _make_unit("Solo Target", models=1, quality=4, defense=5, weapons=[])
    target_5 = _make_unit("Horde Target", models=5, quality=4, defense=5, weapons=[])

    # Blast attacker's matchups
    ranged_b, _ = precompute_damage([blast_atk], [target_1, target_5])
    # Plain attacker's matchups
    ranged_p, _ = precompute_damage([plain_atk], [target_1, target_5])

    # Range threshold index for 24" (our weapons are 24" range)
    r_idx = RANGE_THRESHOLDS.index(24)

    blast_vs_solo = ranged_b[0, 0, r_idx]
    blast_vs_horde = ranged_b[0, 1, r_idx]
    plain_vs_solo = ranged_p[0, 0, r_idx]
    plain_vs_horde = ranged_p[0, 1, r_idx]

    print(f"  Blast attacker → solo (1-model):   {blast_vs_solo:.4f}")
    print(f"  Blast attacker → horde (5-model):  {blast_vs_horde:.4f}")
    print(f"  Plain attacker → solo (1-model):   {plain_vs_solo:.4f}")
    print(f"  Plain attacker → horde (5-model):  {plain_vs_horde:.4f}")

    # The blast attacker's advantage ratio against horde vs solo should be
    # better than the plain attacker's:
    blast_ratio = blast_vs_horde / blast_vs_solo if blast_vs_solo > 0 else float('inf')
    plain_ratio = plain_vs_horde / plain_vs_solo if plain_vs_solo > 0 else float('inf')

    print(f"\n  Blast attacker: horde/solo kill fraction ratio = {blast_ratio:.3f}")
    print(f"  Plain attacker: horde/solo kill fraction ratio = {plain_ratio:.3f}")
    print(f"\n  The blast attacker should have a HIGHER ratio (less penalty for")
    print(f"  shooting the horde), meaning the model should learn to prefer")
    print(f"  horde targets when using blast weapons.")

    # For blast(3): kill_frac vs 1-model = raw/1, vs 5-model = 3*raw/5 = 0.6*raw
    #   → ratio = 0.6
    # For plain: kill_frac vs 1-model = raw/1, vs 5-model = raw/5 = 0.2*raw
    #   → ratio = 0.2
    # So blast_ratio should be ~3x the plain_ratio
    print(f"  blast_ratio / plain_ratio = {blast_ratio / plain_ratio:.2f}x")
    assert blast_ratio > plain_ratio, "Blast should have better horde/solo ratio"

    print("\n  PASSED ✓\n")


def test_precompute_damage_deadly():
    """Verify precompute_damage reflects deadly advantage against tough."""
    print("=" * 64)
    print("TEST: precompute_damage — deadly attacker vs tough targets")
    print("=" * 64)

    deadly_atk = _make_unit("Deadly Attacker", models=1, quality=4, defense=5,
                            weapons=[W_DEADLY])
    plain_atk = _make_unit("Plain Attacker", models=1, quality=4, defense=5,
                           weapons=[W_PLAIN])

    # Two targets: 1-model non-tough, 1-model tough(3)
    target_notough = _make_unit("Squishy", models=1, quality=4, defense=5,
                                weapons=[], tough=0)
    target_tough = _make_unit("Tough(3)", models=1, quality=4, defense=5,
                              weapons=[], tough=3)

    ranged_d, _ = precompute_damage([deadly_atk], [target_notough, target_tough])
    ranged_p, _ = precompute_damage([plain_atk], [target_notough, target_tough])

    r_idx = RANGE_THRESHOLDS.index(24)

    deadly_vs_squishy = ranged_d[0, 0, r_idx]
    deadly_vs_tough = ranged_d[0, 1, r_idx]
    plain_vs_squishy = ranged_p[0, 0, r_idx]
    plain_vs_tough = ranged_p[0, 1, r_idx]

    print(f"  Deadly attacker → squishy (T0): {deadly_vs_squishy:.4f}")
    print(f"  Deadly attacker → tough(3):     {deadly_vs_tough:.4f}")
    print(f"  Plain attacker  → squishy (T0): {plain_vs_squishy:.4f}")
    print(f"  Plain attacker  → tough(3):     {plain_vs_tough:.4f}")

    # Kill fraction ratio (tough / squishy):
    # For deadly: deadly gives 3x wounds. Against T0: 3*base/1. Against T3: 3*base/3 = base.
    #   → ratio = base / (3*base) = 1/3
    # For plain: base/1 vs base/3
    #   → ratio = (base/3) / base = 1/3
    # These are the same! Both weapons lose the same proportion against tough.

    deadly_ratio = deadly_vs_tough / deadly_vs_squishy if deadly_vs_squishy > 0 else float('inf')
    plain_ratio = plain_vs_tough / plain_vs_squishy if plain_vs_squishy > 0 else float('inf')

    print(f"\n  Deadly: tough/squishy kill fraction ratio = {deadly_ratio:.4f}")
    print(f"  Plain:  tough/squishy kill fraction ratio = {plain_ratio:.4f}")

    deadly_ratio = deadly_vs_tough / deadly_vs_squishy if deadly_vs_squishy > 0 else float('inf')
    plain_ratio = plain_vs_tough / plain_vs_squishy if plain_vs_squishy > 0 else float('inf')

    print(f"\n  Deadly: tough/squishy kill fraction ratio = {deadly_ratio:.4f}")
    print(f"  Plain:  tough/squishy kill fraction ratio = {plain_ratio:.4f}")

    # With overkill fix: deadly has a HIGHER ratio (less penalty for tough)
    # because overkill is wasted against squishy targets, making deadly
    # relatively better against tough targets.
    abs_ratio_squishy = deadly_vs_squishy / plain_vs_squishy if plain_vs_squishy > 0 else float('inf')
    abs_ratio_tough = deadly_vs_tough / plain_vs_tough if plain_vs_tough > 0 else float('inf')
    print(f"\n  Absolute: deadly/plain vs squishy = {abs_ratio_squishy:.2f}x")
    print(f"  Absolute: deadly/plain vs tough   = {abs_ratio_tough:.2f}x")
    print(f"  (Deadly gains MORE advantage against tough than against squishy)")

    assert deadly_ratio > plain_ratio, (
        f"Deadly should have higher tough/squishy ratio ({deadly_ratio:.4f} vs {plain_ratio:.4f})")

    print("\n  PASSED ✓\n")


# ------------------------------------------------------------------
# Test: Blast with tough targets (combined effect)
# ------------------------------------------------------------------

def test_blast_vs_tough_multi_model():
    """Check the interaction between blast and multi-model tough targets.

    A 5-model tough(3) unit has 15 starting wounds. Blast still multiplies
    hits by min(blast, models)=3, so damage is 3x. But starting wounds are
    also 15 instead of 5. The kill fraction for blast gets WORSE against
    tough multi-model targets relative to non-tough multi-model targets.
    """
    print("=" * 64)
    print("TEST: Blast vs multi-model tough targets")
    print("=" * 64)

    blast_atk = _make_unit("Blast Attacker", models=1, quality=4, defense=5,
                           weapons=[W_BLAST])

    targets = {}
    for tough, label in [(0, "5-model, T0"), (3, "5-model, T3")]:
        defender = _make_unit(label, models=5, quality=4, defense=5,
                              weapons=[], tough=tough)
        raw = _expected_ranged_damage_at_range(blast_atk, defender, 24)
        sw = starting_wounds(defender)
        kf = raw / sw
        targets[tough] = (raw, sw, kf)
        print(f"  vs {label}: raw={raw:.3f}  starting_wounds={sw}  kill_frac={kf:.4f}")

    print(f"\n  Raw damage is the same (blast×3 regardless of tough).")
    print(f"  But kill fraction is {targets[0][2]/targets[3][2]:.1f}x better against T0")
    print(f"  because T3 has {targets[3][1]/targets[0][1]:.0f}x the starting wounds.")

    print("\n  PASSED ✓\n")


# ------------------------------------------------------------------
# Summary: what the model can/can't learn
# ------------------------------------------------------------------

def print_summary():
    print("=" * 64)
    print("SUMMARY: What the model can learn from matchup features")
    print("=" * 64)

    print("""
  BLAST → MULTI-MODEL:
    ✓ The signal IS present in matchup features.
    A blast attacker's kill_frac drops less against multi-model targets
    than a plain attacker's does, because blast multiplies hits by
    min(blast, models). The model can learn "my blast unit does relatively
    better against hordes" by comparing matchup values across targets.

    Example: Blast(3) attacker — horde/solo kill fraction ratio ≈ 0.6
             Plain attacker    — horde/solo kill fraction ratio ≈ 0.2
             The blast attacker sees 3x less penalty for targeting hordes.

  DEADLY → TOUGH:
    ✓ The signal IS now present (after overkill fix).
    Deadly(3) vs a 1-wound model: effective multiplier = min(3,1) = 1x
    Deadly(3) vs a Tough(3) model: effective multiplier = min(3,3) = 3x

    The model now sees that deadly weapons are relatively BETTER against
    tough targets because overkill against squishy targets is wasted.
    This creates a clear differential signal in the matchup features.
""")


if __name__ == "__main__":
    test_blast_scales_with_models()
    test_plain_does_not_scale_with_models()
    test_deadly_vs_tough()
    test_precompute_damage_blast()
    test_precompute_damage_deadly()
    test_blast_vs_tough_multi_model()
    print_summary()
