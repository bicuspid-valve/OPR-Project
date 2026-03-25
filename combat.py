"""Combat simulator: shooting resolution with per-model range checks, morale."""
from __future__ import annotations

from models import ResolvedUnit, UnitState, _roll, _roll1
from board import dist_sq
import fast_core as _fc


def _effective_defense(defender: UnitState) -> int:
    """Return defender's effective defense, switching to hero defense
    when only the hero model remains."""
    if (defender.hero_unit is not None
            and defender.hero_model_index >= 0
            and not defender._non_hero_alive()):
        return defender.hero_unit.defense
    return defender.unit.defense


# ===================================================================
# TARGETING & DAMAGE SCORING
# ===================================================================

def expected_damage_score(attacker: ResolvedUnit) -> float:
    """Simple proxy for activation ordering: total dice * hit prob."""
    total = 0.0
    hit_prob = (7 - attacker.quality) / 6.0
    for w in attacker.weapons:
        if w.melee:
            continue
        p = 5 / 6 if w.reliable else hit_prob
        total += w.attacks * p * max(1, w.blast) * max(1, w.deadly)
        if w.ap > 0:
            total *= 1.2
    return total


def _precompute_min_dists_sq(a_positions: list[tuple[int, int]],
                             t_positions: list[tuple[int, int]]) -> list[int]:
    """Min squared distance from each attacker model to the nearest target model.

    Computed once and reused for all weapon range checks, avoiding repeated
    O(attacker_models × target_models) passes.
    """
    # --- C-accelerated fast path ---
    if _fc.USE_C_EXT:
        return _fc.fast_min_dists_sq(a_positions, t_positions)

    # --- Pure Python fallback ---
    result = []
    for ac, ar in a_positions:
        best = 999999
        for tc, tr in t_positions:
            dc = ac - tc
            dr = ar - tr
            d = dc * dc + dr * dr
            if d < best:
                best = d
        result.append(best)
    return result


def evaluate_target(attacker: UnitState, target: UnitState) -> tuple[bool, float, bool]:
    """Combined can_shoot_any + base_target_score + is_full_volley in one pass.

    Precomputes per-model minimum distances once, then evaluates all weapon
    range checks against cached distances.

    Returns (can_shoot, damage_score, is_full_volley).
    """
    a_pos = attacker.alive_positions()
    t_pos = target.alive_positions()
    n_a = len(a_pos)

    if n_a == 0 or not t_pos:
        return False, 0.0, True

    min_dists = _precompute_min_dists_sq(a_pos, t_pos)

    # --- can_shoot_any & is_full_volley (per-model weapons) ---
    can_shoot = False
    is_full = True

    for mi in range(min(n_a, len(attacker.weapons_per_model))):
        for w in attacker.weapons_per_model[mi]:
            if w.melee:
                continue
            range_sq = w.range_inches * w.range_inches
            if min_dists[mi] <= range_sq:
                can_shoot = True
            else:
                is_full = False

    if not can_shoot:
        return False, 0.0, False

    # --- Damage score (flat weapon list, same logic as _base_target_score) ---
    hit_prob = (7 - attacker.unit.quality) / 6.0
    expected_wounds = 0.0
    for w in attacker.unit.weapons:
        if w.melee:
            continue
        range_sq = w.range_inches * w.range_inches
        eff = sum(1 for d in min_dists if d <= range_sq)
        if eff == 0:
            continue
        p = 5 / 6 if w.reliable else hit_prob
        hits = w.attacks * p * eff
        if w.blast:
            hits *= min(w.blast, target.models_alive)
        eff_def = min(target.unit.defense + w.ap, 7)
        block_prob = max((7 - eff_def) / 6.0, 1 / 6)
        wounds = hits * (1 - block_prob)
        if w.deadly:
            wounds *= w.deadly
        expected_wounds += wounds

    return can_shoot, expected_wounds / max(target.unit.points, 1), is_full


def models_in_range(attacker: UnitState, target: UnitState,
                    weapon_range: int) -> int:
    """Count attacker models that have at least one target model within range."""
    range_sq = weapon_range * weapon_range
    a_positions = attacker.alive_positions()
    t_positions = target.alive_positions()
    count = 0
    for ac, ar in a_positions:
        for tc, tr in t_positions:
            dc = ac - tc
            dr = ar - tr
            if dc * dc + dr * dr <= range_sq:
                count += 1
                break
    return count


def closest_attacker_distance_sq(attacker: UnitState, target: UnitState) -> int:
    """Squared distance from closest attacker model to nearest target model."""
    a_pos = attacker.alive_positions()
    t_pos = target.alive_positions()
    if not a_pos or not t_pos:
        return 999999
    best = 999999
    for ac, ar in a_pos:
        for tc, tr in t_pos:
            dc = ac - tc
            dr = ar - tr
            d = dc * dc + dr * dr
            if d < best:
                best = d
    return best


def closest_attacker_distance(attacker: UnitState, target: UnitState) -> float:
    """Distance from closest attacker model to nearest target model."""
    import math
    return math.sqrt(closest_attacker_distance_sq(attacker, target))


def can_shoot_any(attacker: UnitState, target: UnitState) -> bool:
    """Check if any attacker model can hit any target model with any ranged weapon."""
    t_positions = target.alive_positions()
    for mi in range(attacker.models_alive):
        mc, mr = attacker.positions[mi]
        for w in attacker.weapons_per_model[mi]:
            if w.melee:
                continue
            range_sq = w.range_inches * w.range_inches
            for tc, tr in t_positions:
                dc = mc - tc
                dr = mr - tr
                if dc * dc + dr * dr <= range_sq:
                    return True
    return False


def is_full_volley(attacker: UnitState, target: UnitState) -> bool:
    """Check if all alive attacker models are in range for all ranged weapons."""
    t_positions = target.alive_positions()
    for mi in range(attacker.models_alive):
        mc, mr = attacker.positions[mi]
        for w in attacker.weapons_per_model[mi]:
            if w.melee:
                continue
            range_sq = w.range_inches * w.range_inches
            in_range = False
            for tc, tr in t_positions:
                dc = mc - tc
                dr = mr - tr
                if dc * dc + dr * dr <= range_sq:
                    in_range = True
                    break
            if not in_range:
                return False
    return True


# ===================================================================
# SHOOTING RESOLUTION
# ===================================================================

def resolve_shooting(attacker: UnitState, defender: UnitState,
                     recorded: bool = False) -> dict | None:
    """Resolve one unit's shooting at one target — with per-model range checks.
    Returns a combat stats dict when recorded=True, or None otherwise."""
    if attacker.shaken:
        attacker.shaken = False
        return None

    d_alive = defender.models_alive
    if d_alive <= 0:
        return None

    # Piercing Spotter
    spotter_ap = 0
    if attacker.unit.piercing_spotter:
        if _roll1() >= 4:
            spotter_ap = 1

    a_quality = attacker.unit.quality
    d_def = _effective_defense(defender)
    if defender.unit.shielded:
        d_def += 1

    # Distance-based modifiers (§5.6, §5.7, §8.6)
    closest_dist_sq = closest_attacker_distance_sq(attacker, defender)
    beyond_9 = closest_dist_sq > 81  # 9.0^2

    # Stealth: defender has stealth and attacker is >9"
    stealth_penalty = 0
    if defender.unit.stealth and beyond_9:
        stealth_penalty = 1

    # Artillery: attacker is artillery and shooting >9" → +1 to hit
    artillery_bonus = 0
    if attacker.unit.artillery and beyond_9:
        artillery_bonus = 1

    # Artillery: defender is artillery and attacker >9" → -2 to hit
    artillery_def_penalty = 0
    if defender.unit.artillery and beyond_9:
        artillery_def_penalty = 2

    # Relentless: attacker has relentless and shooting >9" → nat 6 = extra hit
    relentless_active = attacker.unit.relentless and beyond_9

    # Net hit modifier
    hit_modifier = stealth_penalty + artillery_def_penalty - artillery_bonus

    # Tracking stats
    stat_total_attacks = 0
    stat_total_hits = 0
    stat_total_wounds = 0  # failed defense rolls

    # Per-model weapon iteration: each model fires its own weapons
    d_positions = defender.alive_positions()
    for model_idx in range(attacker.models_alive):
        if defender.models_alive <= 0:
            break

        m_col, m_row = attacker.positions[model_idx]
        model_weapons = attacker.weapons_per_model[model_idx]

        # Group this model's ranged weapons by name (batches dice for
        # single-model units with duplicate weapons like 2x Rapid Shard Cannon)
        weapon_groups: dict[str, tuple] = {}
        for w in model_weapons:
            if w.melee:
                continue
            if w.name in weapon_groups:
                weapon_groups[w.name] = (w, weapon_groups[w.name][1] + 1)
            else:
                weapon_groups[w.name] = (w, 1)

        for weapon, count in weapon_groups.values():
            if defender.models_alive <= 0:
                break

            # Check if this model is in range of any target model
            range_sq = weapon.range_inches * weapon.range_inches
            in_range = False
            for tc, tr in d_positions:
                dc = m_col - tc
                dr = m_row - tr
                if dc * dc + dr * dr <= range_sq:
                    in_range = True
                    break
            if not in_range:
                continue

            total_dice = weapon.attacks * count
            if total_dice == 0:
                continue

            stat_total_attacks += total_dice

            rolls = _roll(total_dice)
            base_thresh = 2 if weapon.reliable else a_quality
            hit_thresh = base_thresh + hit_modifier

            # Classify rolls
            is_nat6 = (rolls == 6)
            is_nat1 = (rolls == 1)
            # Nat 6 always hits, nat 1 always misses
            is_hit = is_nat6 | ((rolls >= hit_thresh) & ~is_nat1)
            nat6_count = int(is_nat6.sum())
            normal_count = int(is_hit.sum()) - nat6_count

            # Relentless extra hits on nat 6
            if relentless_active:
                normal_count += nat6_count  # each nat 6 generates 1 extra hit

            stat_total_hits += nat6_count + normal_count

            if nat6_count == 0 and normal_count == 0:
                continue

            w_ap = weapon.ap
            w_crack = weapon.crack
            w_rending = weapon.rending
            w_blast = weapon.blast
            w_deadly = weapon.deadly
            w_takedown = weapon.takedown
            w_unstoppable = weapon.unstoppable
            w_bane = weapon.bane
            ignore_regen = w_rending or w_unstoppable or w_bane
            is_deadly_weapon = w_deadly > 0 or w_takedown
            deadly_mult = max(w_deadly, 1)

            # AP for nat6 hits
            nat6_ap = w_ap
            if w_crack:
                nat6_ap += 2
            if w_rending:
                nat6_ap += 4

            # Pre-roll defense dice
            total_hits_weapon = nat6_count + normal_count
            max_def_dice = total_hits_weapon * max(w_blast, 1)
            def_rolls = _roll(max_def_dice)
            di = 0

            # --- Fast vectorized defense path (no blast/bane/spotter) ---
            if not w_blast and not w_bane and spotter_ap == 0:
                block_t_nat6 = d_def + nat6_ap
                if block_t_nat6 > 7:
                    block_t_nat6 = 7
                block_t_normal = d_def + w_ap
                if block_t_normal > 7:
                    block_t_normal = 7

                if nat6_count > 0 and defender.models_alive > 0:
                    sl = def_rolls[di:di + nat6_count]
                    wounds = int(((sl != 6) & (sl < block_t_nat6)).sum())
                    di += nat6_count
                    if wounds > 0:
                        if is_deadly_weapon:
                            dealt = defender.apply_deadly_wounds(wounds, deadly_mult,
                                                        ignore_regen, allow_hero=w_takedown)
                        else:
                            dealt = defender.apply_wounds(wounds, ignore_regen,
                                                  allow_hero=w_takedown)
                        stat_total_wounds += dealt

                if normal_count > 0 and defender.models_alive > 0:
                    sl = def_rolls[di:di + normal_count]
                    wounds = int(((sl != 6) & (sl < block_t_normal)).sum())
                    di += normal_count
                    if wounds > 0:
                        if is_deadly_weapon:
                            dealt = defender.apply_deadly_wounds(wounds, deadly_mult,
                                                        ignore_regen, allow_hero=w_takedown)
                        else:
                            dealt = defender.apply_wounds(wounds, ignore_regen,
                                                  allow_hero=w_takedown)
                        stat_total_wounds += dealt
                # Update defender positions cache after casualties
                d_positions = defender.alive_positions()
                continue

            # --- Loop path (blast/bane/spotter) ---
            # Process nat6 hits
            for _ in range(nat6_count):
                if defender.models_alive <= 0:
                    break
                eff_ap = nat6_ap + spotter_ap
                if spotter_ap > 0:
                    spotter_ap = 0
                blast_n = min(w_blast, defender.models_alive) if w_blast else 1

                for _ in range(blast_n):
                    if defender.models_alive <= 0 or di >= max_def_dice:
                        break
                    _dr = int(def_rolls[di]); di += 1
                    # Bane: re-roll nat 6 defense
                    if w_bane and _dr == 6:
                        _dr = _roll1()
                    if _dr == 6:
                        continue
                    block_t = d_def + eff_ap
                    if block_t > 7:
                        block_t = 7
                    if _dr >= block_t:
                        continue
                    stat_total_wounds += 1
                    if is_deadly_weapon:
                        defender.apply_deadly_wounds(1, deadly_mult, ignore_regen,
                                                    allow_hero=w_takedown)
                    else:
                        defender.apply_wounds(1, ignore_regen,
                                              allow_hero=w_takedown)

            # Process normal hits
            for _ in range(normal_count):
                if defender.models_alive <= 0:
                    break
                eff_ap = w_ap + spotter_ap
                if spotter_ap > 0:
                    spotter_ap = 0
                blast_n = min(w_blast, defender.models_alive) if w_blast else 1

                for _ in range(blast_n):
                    if defender.models_alive <= 0 or di >= max_def_dice:
                        break
                    _dr = int(def_rolls[di]); di += 1
                    # Bane: re-roll nat 6 defense
                    if w_bane and _dr == 6:
                        _dr = _roll1()
                    if _dr == 6:
                        continue
                    block_t = d_def + eff_ap
                    if block_t > 7:
                        block_t = 7
                    if _dr >= block_t:
                        continue
                    stat_total_wounds += 1
                    if is_deadly_weapon:
                        defender.apply_deadly_wounds(1, deadly_mult, ignore_regen,
                                                    allow_hero=w_takedown)
                    else:
                        defender.apply_wounds(1, ignore_regen,
                                              allow_hero=w_takedown)

            # Update defender positions cache after casualties
            d_positions = defender.alive_positions()

    if stat_total_attacks == 0:
        return None

    if not recorded:
        return None  # side effects (wounds) already applied

    # Build weapon summary for display (counted, with full stats)
    weapon_counts: dict[str, int] = {}
    weapon_info: dict[str, dict] = {}
    for mw in attacker.weapons_per_model:
        for w in mw:
            if w.melee:
                continue
            weapon_counts[w.name] = weapon_counts.get(w.name, 0) + 1
            if w.name not in weapon_info:
                abilities = []
                if w.ap:
                    abilities.append(f"AP({w.ap})")
                if w.blast:
                    abilities.append(f"Blast({w.blast})")
                if w.deadly:
                    abilities.append(f"Deadly({w.deadly})")
                if w.crack:
                    abilities.append("Crack")
                if w.rending:
                    abilities.append("Rending")
                if w.reliable:
                    abilities.append("Reliable")
                if w.takedown:
                    abilities.append("Takedown")
                if w.unstoppable:
                    abilities.append("Unstoppable")
                weapon_info[w.name] = {
                    'name': w.name,
                    'range': w.range_inches,
                    'attacks': w.attacks,
                    'abilities': abilities,
                }
    weapon_details = []
    for name in weapon_counts:
        entry = dict(weapon_info[name])
        entry['count'] = weapon_counts[name]
        weapon_details.append(entry)

    # Collect unit-level special rules
    attacker_rules = []
    au = attacker.unit
    if au.scout:
        attacker_rules.append("Scout")
    if au.stealth:
        attacker_rules.append("Stealth")
    if au.relentless:
        attacker_rules.append("Relentless")
    if au.fast:
        attacker_rules.append("Fast")
    if au.artillery:
        attacker_rules.append("Artillery")
    if au.fearless:
        attacker_rules.append("Fearless")
    if au.regeneration:
        attacker_rules.append("Regeneration")
    if au.piercing_spotter:
        attacker_rules.append("Piercing Spotter")
    if au.tough:
        attacker_rules.append(f"Tough({au.tough})")
    if au.shielded:
        attacker_rules.append("Shielded")
    if au.furious:
        attacker_rules.append("Furious")
    if au.impact:
        attacker_rules.append(f"Impact({au.impact})")
    if au.fear:
        attacker_rules.append(f"Fear({au.fear})")

    defender_rules = []
    du = defender.unit
    if du.scout:
        defender_rules.append("Scout")
    if du.stealth:
        defender_rules.append("Stealth")
    if du.relentless:
        defender_rules.append("Relentless")
    if du.fast:
        defender_rules.append("Fast")
    if du.artillery:
        defender_rules.append("Artillery")
    if du.fearless:
        defender_rules.append("Fearless")
    if du.regeneration:
        defender_rules.append("Regeneration")
    if du.tough:
        defender_rules.append(f"Tough({du.tough})")
    if du.shielded:
        defender_rules.append("Shielded")
    if du.fortified:
        defender_rules.append("Fortified")

    return {
        'attacker_quality': a_quality,
        'attacker_weapons': weapon_details,
        'attacker_rules': attacker_rules,
        'defender_defense': d_def,
        'defender_rules': defender_rules,
        'total_attacks': stat_total_attacks,
        'total_hits': stat_total_hits,
        'total_wounds': stat_total_wounds,
        'hit_modifier': hit_modifier,
    }


# ===================================================================
# MORALE
# ===================================================================

def check_morale(unit: UnitState):
    if unit.models_alive <= 0 or unit.morale_checked:
        return
    if not unit.at_half_or_below():
        return
    unit.morale_checked = True
    quality = unit.unit.quality
    # Hero quality override (use better quality)
    if unit.hero_unit is not None and unit.hero_model_index >= 0:
        quality = min(quality, unit.hero_unit.quality)
    passed = _roll1() >= quality
    if not passed and unit.unit.fearless:
        passed = _roll1() >= 4
    if not passed:
        unit.shaken = True


# ===================================================================
# MELEE COMBAT
# ===================================================================

MELEE_RANGE_SQ = 4  # 2 squares c2c
SIMPLE_MELEE = True  # skip per-model range checks; all alive models attack


def models_in_melee_range(attacker: UnitState,
                          defender: UnitState) -> int:
    """Count attacker models within 2\" c2c of any defender model."""
    return models_in_range(attacker, defender, 2)


def resolve_melee(attacker: UnitState, defender: UnitState,
                  is_charge: bool = False,
                  is_strike_back: bool = False,
                  recorded: bool = False) -> dict | int | None:
    """Resolve one unit's melee attacks against one target.
    When recorded=True, returns a full combat stats dict.
    When recorded=False, returns wounds_dealt as int (or None if no melee)."""
    if defender.models_alive <= 0:
        return None

    a_quality = attacker.unit.quality
    d_def = _effective_defense(defender)
    if defender.unit.shielded:
        d_def += 1

    # Fatigue: only nat 6 hits
    fatigued = getattr(attacker, 'fatigued', False)

    # Furious: extra hits on nat 6 during charge
    furious_active = is_charge and attacker.unit.furious

    stat_total_attacks = 0
    stat_total_hits = 0
    stat_total_wounds = 0

    # Per-model melee weapon iteration
    melee_range_sq = MELEE_RANGE_SQ
    has_any_melee = False

    for model_idx in range(attacker.models_alive):
        if defender.models_alive <= 0:
            break

        m_col, m_row = attacker.positions[model_idx]
        model_weapons = attacker.weapons_per_model[model_idx]

        # Check if this model is in melee range of any defender model
        if not SIMPLE_MELEE:
            in_range = False
            for tc, tr in defender.alive_positions():
                dc = m_col - tc
                dr = m_row - tr
                if dc * dc + dr * dr <= melee_range_sq:
                    in_range = True
                    break
            if not in_range:
                continue

        # Group this model's melee weapons by name
        weapon_groups: dict[str, tuple] = {}
        for w in model_weapons:
            if not w.melee:
                continue
            if w.name in weapon_groups:
                weapon_groups[w.name] = (w, weapon_groups[w.name][1] + 1)
            else:
                weapon_groups[w.name] = (w, 1)

        if not weapon_groups:
            continue
        has_any_melee = True

        for weapon, count in weapon_groups.values():
            if defender.models_alive <= 0:
                break

            total_dice = weapon.attacks * count
            if total_dice == 0:
                continue

            stat_total_attacks += total_dice

            rolls = _roll(total_dice)

            # Thrust: +1 to hit and +1 AP when charging
            thrust_active = is_charge and weapon.thrust

            if fatigued:
                hit_thresh = 6  # only nat 6 hits when fatigued
            else:
                base_qual = 2 if weapon.reliable else a_quality
                hit_thresh = max(base_qual - (1 if thrust_active else 0), 2)

            is_nat6 = (rolls == 6)
            is_nat1 = (rolls == 1)
            is_hit = is_nat6 | ((rolls >= hit_thresh) & ~is_nat1)
            nat6_count = int(is_nat6.sum())
            normal_count = int(is_hit.sum()) - nat6_count

            # Furious extra hits on nat 6
            if furious_active:
                normal_count += nat6_count

            stat_total_hits += nat6_count + normal_count

            if nat6_count == 0 and normal_count == 0:
                continue

            w_ap = weapon.ap + (1 if thrust_active else 0)
            # Fortified: reduce AP by 1 (min 0)
            if defender.unit.fortified:
                w_ap = max(w_ap - 1, 0)
            w_crack = weapon.crack
            w_rending = weapon.rending
            w_blast = weapon.blast
            w_deadly = weapon.deadly
            w_bane = weapon.bane
            w_unstoppable = weapon.unstoppable
            ignore_regen = w_rending or w_unstoppable or w_bane
            is_deadly_weapon = w_deadly > 0
            deadly_mult = max(w_deadly, 1)

            # AP for nat6 hits
            nat6_ap = w_ap
            if w_crack:
                nat6_ap += 2
            if w_rending:
                nat6_ap += 4

            # Pre-roll defense dice
            total_hits_weapon = nat6_count + normal_count
            max_def_dice = total_hits_weapon * max(w_blast, 1)
            def_rolls = _roll(max_def_dice)
            di = 0

            # --- Fast vectorized defense path (no blast/bane) ---
            if not w_blast and not w_bane:
                block_t_nat6 = d_def + nat6_ap
                if block_t_nat6 > 7:
                    block_t_nat6 = 7
                block_t_normal = d_def + w_ap
                if block_t_normal > 7:
                    block_t_normal = 7

                if nat6_count > 0 and defender.models_alive > 0:
                    sl = def_rolls[di:di + nat6_count]
                    wounds = int(((sl != 6) & (sl < block_t_nat6)).sum())
                    di += nat6_count
                    stat_total_wounds += wounds
                    if wounds > 0:
                        if is_deadly_weapon:
                            defender.apply_deadly_wounds(wounds, deadly_mult,
                                                        ignore_regen)
                        else:
                            defender.apply_wounds(wounds, ignore_regen)

                if normal_count > 0 and defender.models_alive > 0:
                    sl = def_rolls[di:di + normal_count]
                    wounds = int(((sl != 6) & (sl < block_t_normal)).sum())
                    di += normal_count
                    stat_total_wounds += wounds
                    if wounds > 0:
                        if is_deadly_weapon:
                            defender.apply_deadly_wounds(wounds, deadly_mult,
                                                        ignore_regen)
                        else:
                            defender.apply_wounds(wounds, ignore_regen)
                continue

            # --- Loop path (blast/bane) ---
            # Process nat6 hits
            for _ in range(nat6_count):
                if defender.models_alive <= 0:
                    break
                eff_ap = nat6_ap
                blast_n = min(w_blast, defender.models_alive) if w_blast else 1

                for _ in range(blast_n):
                    if defender.models_alive <= 0 or di >= max_def_dice:
                        break
                    _dr = int(def_rolls[di]); di += 1
                    if w_bane and _dr == 6:
                        _dr = _roll1()
                    if _dr == 6:
                        continue
                    block_t = d_def + eff_ap
                    if block_t > 7:
                        block_t = 7
                    if _dr >= block_t:
                        continue
                    stat_total_wounds += 1
                    if is_deadly_weapon:
                        defender.apply_deadly_wounds(1, deadly_mult, ignore_regen)
                    else:
                        defender.apply_wounds(1, ignore_regen)

            # Process normal hits
            for _ in range(normal_count):
                if defender.models_alive <= 0:
                    break
                eff_ap = w_ap
                blast_n = min(w_blast, defender.models_alive) if w_blast else 1

                for _ in range(blast_n):
                    if defender.models_alive <= 0 or di >= max_def_dice:
                        break
                    _dr = int(def_rolls[di]); di += 1
                    if w_bane and _dr == 6:
                        _dr = _roll1()
                    if _dr == 6:
                        continue
                    block_t = d_def + eff_ap
                    if block_t > 7:
                        block_t = 7
                    if _dr >= block_t:
                        continue
                    stat_total_wounds += 1
                    if is_deadly_weapon:
                        defender.apply_deadly_wounds(1, deadly_mult, ignore_regen)
                    else:
                        defender.apply_wounds(1, ignore_regen)

    if stat_total_attacks == 0:
        return None

    if not recorded:
        return stat_total_wounds

    # Build weapon summary for display
    weapon_counts: dict[str, int] = {}
    weapon_info: dict[str, dict] = {}
    for mw in attacker.weapons_per_model:
        for w in mw:
            if not w.melee:
                continue
            weapon_counts[w.name] = weapon_counts.get(w.name, 0) + 1
            if w.name not in weapon_info:
                abilities = []
                if w.ap:
                    abilities.append(f"AP({w.ap})")
                if w.blast:
                    abilities.append(f"Blast({w.blast})")
                if w.deadly:
                    abilities.append(f"Deadly({w.deadly})")
                if w.crack:
                    abilities.append("Crack")
                if w.rending:
                    abilities.append("Rending")
                if w.reliable:
                    abilities.append("Reliable")
                if w.bane:
                    abilities.append("Bane")
                abilities.append("Melee")
                weapon_info[w.name] = {
                    'name': w.name,
                    'range': 0,
                    'attacks': w.attacks,
                    'abilities': abilities,
                }
    weapon_details = []
    for name in weapon_counts:
        entry = dict(weapon_info[name])
        entry['count'] = weapon_counts[name]
        weapon_details.append(entry)

    return {
        'combat_type': 'melee',
        'attacker_quality': a_quality,
        'attacker_weapons': weapon_details,
        'attacker_rules': [],
        'defender_defense': d_def,
        'defender_rules': [],
        'total_attacks': stat_total_attacks,
        'total_hits': stat_total_hits,
        'total_wounds': stat_total_wounds,
        'wounds_dealt': stat_total_wounds,
        'hit_modifier': 0,
        'is_charge': is_charge,
        'is_strike_back': is_strike_back,
        'fatigued': fatigued,
    }


# ===================================================================
# IMPACT
# ===================================================================

def resolve_impact(attacker: UnitState, defender: UnitState) -> dict:
    """Resolve Impact(X) hits on charge. Each model in range rolls X dice, 2+ = hit.
    Returns {'impact_hits': int, 'impact_wounds': int}."""
    impact_val = attacker.unit.impact
    if impact_val <= 0 or defender.models_alive <= 0:
        return {'impact_hits': 0, 'impact_wounds': 0}

    d_def = _effective_defense(defender)
    if defender.unit.shielded:
        d_def += 1

    # Count attacker models in melee range
    if SIMPLE_MELEE:
        in_range = attacker.models_alive
    else:
        in_range = models_in_melee_range(attacker, defender)
    if in_range == 0:
        return {'impact_hits': 0, 'impact_wounds': 0}

    total_dice = impact_val * in_range
    rolls = _roll(total_dice)
    hits = int((rolls >= 2).sum())

    if hits == 0:
        return {'impact_hits': 0, 'impact_wounds': 0}

    # Resolve hits against defender's defense (no AP)
    def_rolls = _roll(hits)
    wounds = 0
    for i in range(hits):
        if defender.models_alive <= 0:
            break
        dr = int(def_rolls[i])
        if dr == 6:
            continue
        if dr >= d_def:
            continue
        wounds += 1
        defender.apply_wounds(1)

    return {'impact_hits': hits, 'impact_wounds': wounds}


# ===================================================================
# MELEE MORALE
# ===================================================================

def check_melee_morale(unit: UnitState, wounds_dealt: int,
                       wounds_received: int):
    """Melee morale: loser (dealt fewer wounds) takes quality test.
    Fear(X) adds X bonus wounds to the unit's dealt total for comparison.
    Failed + at half or below = rout (destroyed).
    Failed above half = Shaken. Fearless: re-roll on 4+.
    Equal wounds = no test."""
    if unit.models_alive <= 0:
        return
    effective_dealt = wounds_dealt + unit.unit.fear
    if effective_dealt >= wounds_received:
        return  # won or tied, no test

    quality = unit.unit.quality
    # Hero quality override (use better quality)
    hero_unit = getattr(unit, 'hero_unit', None)
    if hero_unit is not None:
        quality = min(quality, hero_unit.quality)

    passed = _roll1() >= quality
    if not passed and unit.unit.fearless:
        passed = _roll1() >= 4
    if not passed:
        if unit.at_half_or_below():
            # Rout — destroyed
            unit.rout()
        else:
            unit.shaken = True


# ===================================================================
# MELEE DAMAGE SCORING (for AI)
# ===================================================================

def expected_melee_damage_score(attacker: ResolvedUnit) -> float:
    """Proxy for melee activation ordering: total melee dice * hit prob."""
    total = 0.0
    hit_prob = (7 - attacker.quality) / 6.0
    for w in attacker.weapons:
        if not w.melee:
            continue
        p = 5 / 6 if w.reliable else hit_prob
        total += w.attacks * p * max(1, w.blast) * max(1, w.deadly)
        if w.ap > 0:
            total *= 1.2
    return total
