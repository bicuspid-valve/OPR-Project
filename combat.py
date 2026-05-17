"""Combat simulator: shooting resolution with per-model range checks, morale."""
from __future__ import annotations

from models import ResolvedUnit, UnitState, _roll, _roll1
from board import Board, dist_sq
import fast_core as _fc


def _effective_defense(defender: UnitState, cover_bonus: int = 0) -> int:
    """Return defender's effective defense save threshold, lowered by
    ``cover_bonus`` (TERRAIN_SPEC.md §4.4 cover gives +1 to defense rolls,
    which is equivalent to a -1 to the save threshold). Hero defense applies
    when only the hero model remains. Callers compute ``cover_bonus`` per
    shooter / per weapon (suppressed by ``ignores_cover``).
    """
    if (defender.hero_unit is not None
            and defender.hero_model_index >= 0
            and not defender._non_hero_alive()):
        return defender.hero_unit.defense - cover_bonus
    return defender.unit.defense - cover_bonus


def _shooter_cover_lookup(board: Board | None, Y_sq: tuple[int, int],
                          target_squares: list[tuple[int, int]]
                          ) -> tuple[int, list[bool]]:
    """Per-shooter visibility/cover aggregator over a defender unit's models.

    Returns ``(n_cover_or_invisible, visible_mask)``. When the precomputed
    visibility/cover table is attached to the board (``board.vis_cover_table``),
    the lookup uses it; otherwise falls back to live geometric compute via
    :mod:`terrain_los`. Empty terrain → all visible, none in cover.
    """
    if board is None or not board.terrain:
        return 0, [True] * len(target_squares)
    table = getattr(board, "vis_cover_table", None)
    if table is not None:
        from vis_cover_table import OPEN, COVER, NO_LOS, lookup as _table_lookup
        n_bad = 0
        visible = [False] * len(target_squares)
        for i, A_sq in enumerate(target_squares):
            v = _table_lookup(table, Y_sq, A_sq)
            if v == NO_LOS:
                n_bad += 1
            else:
                visible[i] = True
                if v == COVER:
                    n_bad += 1
        return n_bad, visible
    from terrain_los import shooter_cover_state
    return shooter_cover_state(Y_sq, target_squares, board.terrain)


def _eff_range_sq(weapon, attacker_unit: ResolvedUnit) -> int:
    """Squared range for `weapon` fired from `attacker_unit`.

    Versatile Reach grants +4" range to all the unit's ranged weapons. Melee
    weapons (range 0) are unaffected.
    """
    rng = weapon.range_inches
    if rng > 0 and attacker_unit.versatile_reach:
        rng += 4
    return rng * rng


def _smash_active(weapon, defender: UnitState) -> bool:
    """Smash gains Blast(+3) when more than half of the target's models have
    Defense 5+ (i.e. defense stat >= 5). 'Most' = strictly more than half per
    the user's interpretation."""
    if not weapon.smash:
        return False
    n = defender.models_alive
    if n <= 0:
        return False
    weak_count = 0
    if defender.unit.defense >= 5:
        weak_count = n
    return weak_count * 2 > n


def _tear_active(weapon, defender: UnitState) -> bool:
    """Tear: AP(+4) against units where most models have Tough(9)+. Tough
    here is the model's Tough stat; we approximate by checking the unit's
    Tough value against models_alive."""
    if not weapon.tear:
        return False
    n = defender.models_alive
    if n <= 0:
        return False
    # All models in our model share unit.tough — uniform-tough assumption.
    if defender.unit.tough >= 9:
        return True
    return False


def _puncture_active(weapon, defender: UnitState) -> bool:
    """Puncture: ignores Regen always; AP(+4) against units where most models
    have Tough(3) to Tough(9) inclusive."""
    if not weapon.puncture:
        return False
    return 3 <= defender.unit.tough <= 9


def _versatile_attack_pick(attacker_unit: ResolvedUnit, defender: UnitState,
                           quality: int, base_modifier: int) -> str:
    """Pick AP(+1) vs +1-to-hit by aggregate expected wounds across weapons.

    Returns 'ap' or 'hit'. Used only for shooting/charging enemies > 9" away
    where Versatile Attack actually triggers.
    """
    d_def = defender.unit.defense
    if defender.unit.shielded:
        d_def += 1

    ev_ap = 0.0
    ev_hit = 0.0
    for w in attacker_unit.weapons:
        if w.melee:
            continue
        a = w.attacks
        base_thresh = 2 if w.reliable else quality
        deadly_mult = max(1, w.deadly)
        # Option AP: +1 AP across the engagement
        thresh = max(2, min(6, base_thresh + base_modifier))
        p_hit = max(0.0, (7 - thresh)) / 6.0
        block = min(7, d_def + w.ap + 1)
        p_block = max(0.0, (7 - block)) / 6.0
        ev_ap += a * p_hit * (1.0 - p_block) * deadly_mult
        # Option Hit: -1 to-hit (effective threshold reduced by 1)
        thresh_h = max(2, min(6, base_thresh + base_modifier - 1))
        p_hit_h = max(0.0, (7 - thresh_h)) / 6.0
        block_h = min(7, d_def + w.ap)
        p_block_h = max(0.0, (7 - block_h)) / 6.0
        ev_hit += a * p_hit_h * (1.0 - p_block_h) * deadly_mult
    return 'ap' if ev_ap > ev_hit else 'hit'


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


def evaluate_target(attacker: UnitState, target: UnitState,
                    board: Board | None = None) -> tuple[bool, float, bool]:
    """Combined can_shoot_any + base_target_score + is_full_volley in one pass.

    Precomputes per-model minimum distances once, then evaluates all weapon
    range checks against cached distances. When ``board`` carries terrain,
    a defender model only counts as a valid target for a given shooter when
    it is visible from that shooter (TERRAIN_SPEC.md §4.4(1)).

    Returns (can_shoot, damage_score, is_full_volley).
    """
    a_pos = attacker.alive_positions()
    t_pos = target.alive_positions()
    n_a = len(a_pos)
    n_t = len(t_pos)

    if n_a == 0 or not t_pos:
        return False, 0.0, True

    has_terrain = board is not None and bool(board.terrain)

    if has_terrain:
        # Per-attacker-model visibility + full distance matrix (n_a x n_t).
        # Visibility forces us to evaluate per (shooter, target-model) pair
        # rather than relying on the precomputed nearest-defender distance.
        vis_masks: list[list[bool]] = []
        dists_per_mi: list[list[int]] = []
        for ac, ar in a_pos:
            _, vmask = _shooter_cover_lookup(board, (ac, ar), t_pos)
            vis_masks.append(vmask)
            row = []
            for tc, tr in t_pos:
                dc = ac - tc
                dr = ar - tr
                row.append(dc * dc + dr * dr)
            dists_per_mi.append(row)
        min_dists = None
    else:
        vis_masks = None
        dists_per_mi = None
        min_dists = _precompute_min_dists_sq(a_pos, t_pos)

    def _shooter_in_range(mi: int, range_sq: int) -> bool:
        if has_terrain:
            vmask = vis_masks[mi]
            dists = dists_per_mi[mi]
            for ti in range(n_t):
                if vmask[ti] and dists[ti] <= range_sq:
                    return True
            return False
        return min_dists[mi] <= range_sq

    # --- can_shoot_any & is_full_volley (per-model weapons) ---
    can_shoot = False
    is_full = True

    for mi in range(min(n_a, len(attacker.weapons_per_model))):
        for w in attacker.weapons_per_model[mi]:
            if w.melee:
                continue
            range_sq = _eff_range_sq(w, attacker.unit)
            if _shooter_in_range(mi, range_sq):
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
        range_sq = _eff_range_sq(w, attacker.unit)
        if has_terrain:
            eff = sum(1 for mi in range(n_a) if _shooter_in_range(mi, range_sq))
        else:
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


def can_shoot_any(attacker: UnitState, target: UnitState,
                  board: Board | None = None) -> bool:
    """Check if any attacker model can hit any target model with any ranged weapon.

    When ``board`` carries terrain, the visibility constraint of
    TERRAIN_SPEC.md §4.4(1) also applies: at least one (attacker model, target
    model) pair must be mutually visible.
    """
    t_positions = target.alive_positions()
    has_terrain = board is not None and bool(board.terrain)
    for mi in range(attacker.models_alive):
        mc, mr = attacker.positions[mi]
        for w in attacker.weapons_per_model[mi]:
            if w.melee:
                continue
            range_sq = _eff_range_sq(w, attacker.unit)
            for tc, tr in t_positions:
                dc = mc - tc
                dr = mr - tr
                if dc * dc + dr * dr > range_sq:
                    continue
                if not has_terrain:
                    return True
                # Visibility gate (only used when terrain is present).
                from terrain_los import is_visible
                if is_visible((mc, mr), (tc, tr), board.terrain):
                    return True
    return False


def is_full_volley(attacker: UnitState, target: UnitState,
                   board: Board | None = None) -> bool:
    """Check if all alive attacker models are in range for all ranged weapons.

    When ``board`` carries terrain, the in-range defender must also be visible
    from the shooter (TERRAIN_SPEC.md §4.4(1)).
    """
    t_positions = target.alive_positions()
    has_terrain = board is not None and bool(board.terrain)
    vis_masks: list[list[bool]] | None = None
    if has_terrain:
        vis_masks = [
            _shooter_cover_lookup(board, attacker.positions[mi], t_positions)[1]
            for mi in range(attacker.models_alive)
        ]
    for mi in range(attacker.models_alive):
        mc, mr = attacker.positions[mi]
        vmask = vis_masks[mi] if vis_masks is not None else None
        for w in attacker.weapons_per_model[mi]:
            if w.melee:
                continue
            range_sq = _eff_range_sq(w, attacker.unit)
            in_range = False
            for ti, (tc, tr) in enumerate(t_positions):
                if vmask is not None and not vmask[ti]:
                    continue
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
                     recorded: bool = False,
                     board: Board | None = None) -> dict | None:
    """Resolve one unit's shooting at one target — with per-model range checks.

    When ``board`` carries terrain, per-shooter visibility and cover (per
    TERRAIN_SPEC.md §4.4) gate firing and add a +1 defense bonus to the
    target on a per-shooter majority basis.

    Returns a combat stats dict when recorded=True, or None otherwise.
    """
    if attacker.shaken:
        attacker.shaken = False
        return None

    d_alive = defender.models_alive
    if d_alive <= 0:
        return None

    has_terrain = board is not None and bool(board.terrain)

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

    # Versatile Attack — applies only to enemies > 9" away. Pick AP(+1) or
    # +1-to-hit based on which yields more expected damage (per the user's
    # specification). The chosen mode is applied to all weapons in the
    # engagement.
    va_ap_bonus = 0
    if attacker.unit.versatile_attack and beyond_9:
        mode = _versatile_attack_pick(attacker.unit, defender, a_quality, hit_modifier)
        if mode == 'ap':
            va_ap_bonus = 1
        else:
            hit_modifier -= 1

    # Unstoppable Mark: target has been marked → all attackers' weapons get
    # Unstoppable for this engagement (ignores Regen). The mark is one-shot.
    mark_force_unstoppable = bool(defender.marked_by_unstoppable)
    if mark_force_unstoppable:
        defender.marked_by_unstoppable = False

    # ED: Ignores Cover — strip the Stealth +1-to-hit penalty for the whole unit.
    if attacker.unit.ignores_cover and stealth_penalty:
        hit_modifier -= stealth_penalty
        stealth_penalty = 0  # downstream Indirect-vs-stealth check is now a no-op

    # ED: Piercing Hunter — +1 AP to all this unit's weapons when shooting >9".
    ph_ap_bonus = 1 if (attacker.unit.piercing_hunter and beyond_9) else 0

    # ED: Increased Shooting Range Mark — friendly side gets +6" range when
    # shooting at the marked target (consumed by the first attack).
    isr_range_bonus = 0
    if defender.marked_by_isr and attacker.owner == defender.marked_by_isr_owner:
        isr_range_bonus = 6
        defender.marked_by_isr = False
        defender.marked_by_isr_owner = ""

    # ED: Vengeance — attacker gets +X to hit when target carries X markers
    # placed by their faction's destroyed Vengeance units.
    if defender.vengeance_markers:
        hit_modifier -= defender.vengeance_markers

    # ED: Clan Warrior (army-wide) — per unmodified roll of 6 to hit (5+ with
    # Clan Warrior Boost), this model rolls one extra attack with that weapon.
    # The newly-generated attacks do NOT recurse.
    cw_active = attacker.unit.clan_warrior
    cw_thresh = 5 if attacker.unit.clan_warrior_boost else 6

    # Tracking stats
    stat_total_attacks = 0
    stat_total_hits = 0
    stat_total_wounds = 0  # failed defense rolls

    unit_ignores_cover = bool(attacker.unit.ignores_cover)

    # Per-model weapon iteration: each model fires its own weapons
    d_positions = defender.alive_positions()
    for model_idx in range(attacker.models_alive):
        if defender.models_alive <= 0:
            break

        m_col, m_row = attacker.positions[model_idx]
        model_weapons = attacker.weapons_per_model[model_idx]

        # Per-shooter cover/visibility state (TERRAIN_SPEC.md §4.4).
        # When terrain is empty, this is a no-op: all visible, no cover.
        if has_terrain:
            n_bad, vis_mask = _shooter_cover_lookup(
                board, (m_col, m_row), d_positions)
            if not any(vis_mask):
                continue  # this shooter sees no defender model
            cover_bonus_unit = 1 if (2 * n_bad > defender.models_alive) else 0
        else:
            cover_bonus_unit = 0

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

            # Per-weapon cover bonus — suppressed by per-weapon or unit-level
            # ignores_cover (§4.5). Always 0 when no terrain.
            if cover_bonus_unit and not (weapon.ignores_cover or unit_ignores_cover):
                eff_cover = 1
            else:
                eff_cover = 0

            # Check if this model is in range of any target model. ISR Mark
            # adds +6" against the marked target.
            range_sq_base = _eff_range_sq(weapon, attacker.unit)
            if isr_range_bonus and weapon.range_inches > 0:
                eff_rng = weapon.range_inches + (4 if attacker.unit.versatile_reach else 0) + isr_range_bonus
                range_sq = eff_rng * eff_rng
            else:
                range_sq = range_sq_base
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
            # Indirect: ignores cover. We don't model cover explicitly, but the
            # closest analog is the Stealth +1-to-hit penalty applied at >9".
            # An Indirect weapon strips that component.
            w_hit_modifier = hit_modifier
            if weapon.indirect and stealth_penalty:
                w_hit_modifier -= stealth_penalty
            hit_thresh = base_thresh + w_hit_modifier

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

            # Surge (per-weapon): each nat 6 yields +1 extra hit. Surge stacks
            # with Relentless when both apply — both add an extra normal hit.
            if weapon.surge:
                normal_count += nat6_count

            # Clan Warrior: per nat 6 (or 5+ with Boost) the model rolls one
            # extra attack with this weapon. The bonus rolls do not recurse.
            if cw_active:
                if cw_thresh == 6:
                    bonus_dice = nat6_count
                else:
                    bonus_dice = int((rolls >= cw_thresh).sum())
                if bonus_dice > 0:
                    bonus_rolls = _roll(bonus_dice)
                    b_is_nat6 = (bonus_rolls == 6)
                    b_is_nat1 = (bonus_rolls == 1)
                    b_is_hit = b_is_nat6 | ((bonus_rolls >= hit_thresh) & ~b_is_nat1)
                    b_nat6 = int(b_is_nat6.sum())
                    b_normal = int(b_is_hit.sum()) - b_nat6
                    nat6_count += b_nat6
                    normal_count += b_normal
                    stat_total_attacks += bonus_dice

            stat_total_hits += nat6_count + normal_count

            if nat6_count == 0 and normal_count == 0:
                continue

            w_ap = weapon.ap + va_ap_bonus + ph_ap_bonus  # +VA AP, +Piercing Hunter
            # ED: Tear → +4 AP vs Tough(9)+ majority.
            #     Puncture → +4 AP vs Tough(3)–(9), and ignores Regen.
            w_tear_active = _tear_active(weapon, defender)
            w_puncture_active = _puncture_active(weapon, defender)
            if w_tear_active:
                w_ap += 4
            if w_puncture_active:
                w_ap += 4
            w_crack = weapon.crack
            w_rending = weapon.rending
            w_blast = weapon.blast
            # Smash: against units where >50% of models have Defense 5+,
            # the weapon gains Blast(+3). Smash also ignores Regen.
            w_smash_active = _smash_active(weapon, defender)
            if w_smash_active:
                w_blast = (w_blast or 0) + 3
            w_deadly = weapon.deadly
            w_takedown = weapon.takedown
            w_unstoppable = weapon.unstoppable or mark_force_unstoppable
            # Bane when Shooting Aura: every ranged weapon picks up Bane.
            w_bane = weapon.bane or attacker.unit.bane_shoot
            w_lacerate = weapon.lacerate
            w_shred = weapon.shred
            ignore_regen = (w_rending or w_unstoppable or w_bane
                            or w_smash_active or w_puncture_active)
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

            # --- Fast vectorized defense path (no blast/bane/spotter/lacerate/shred) ---
            if (not w_blast and not w_bane and spotter_ap == 0
                    and not w_lacerate and not w_shred):
                block_t_nat6 = d_def + nat6_ap - eff_cover
                if block_t_nat6 > 7:
                    block_t_nat6 = 7
                block_t_normal = d_def + w_ap - eff_cover
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

            # --- Loop path (blast/bane/spotter/lacerate/shred) ---
            def _resolve_one(def_idx: int, eff_ap: int):
                """Resolve a single defense die. Returns updated def_idx."""
                nonlocal stat_total_wounds
                if def_idx >= max_def_dice or defender.models_alive <= 0:
                    return def_idx
                _dr = int(def_rolls[def_idx]); def_idx += 1
                # Bane: re-roll nat 6 defense
                if w_bane and _dr == 6:
                    _dr = _roll1()
                if _dr == 6:
                    return def_idx
                block_t = d_def + eff_ap - eff_cover
                if block_t > 7:
                    block_t = 7
                # Lacerate: defender re-rolls successful blocks (unmodified).
                # Use the new die as the final result; Bane still re-rolls a
                # natural 6 if that's the new face.
                if w_lacerate and _dr >= block_t:
                    _dr = _roll1()
                    if w_bane and _dr == 6:
                        _dr = _roll1()
                    if _dr == 6:
                        return def_idx
                if _dr >= block_t:
                    return def_idx
                # The hit goes through.
                stat_total_wounds += 1
                if is_deadly_weapon:
                    defender.apply_deadly_wounds(1, deadly_mult, ignore_regen,
                                                allow_hero=w_takedown)
                else:
                    defender.apply_wounds(1, ignore_regen,
                                          allow_hero=w_takedown)
                # Shred: unmodified Defense roll of 1 → +1 raw wound.
                if w_shred and _dr == 1 and defender.models_alive > 0:
                    stat_total_wounds += 1
                    defender.apply_wounds(1, ignore_regen,
                                          allow_hero=w_takedown)
                return def_idx

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
                    di = _resolve_one(di, eff_ap)

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
                    di = _resolve_one(di, eff_ap)

    # ED Vengeance: if the defender was just destroyed and had Vengeance,
    # mark the attacker with as many markers as the defender's starting size.
    if (defender.unit.vengeance and defender.models_alive <= 0
            and getattr(defender, "_vengeance_paid", False) is False):
        defender._vengeance_paid = True
        attacker.vengeance_markers += defender.unit.models

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
                if w.lacerate:
                    abilities.append("Lacerate")
                if w.shred:
                    abilities.append("Shred")
                if w.smash:
                    abilities.append("Smash")
                if w.indirect:
                    abilities.append("Indirect")
                if w.limited:
                    abilities.append("Limited")
                if w.surge:
                    abilities.append("Surge")
                if w.tear:
                    abilities.append("Tear")
                if w.puncture:
                    abilities.append("Puncture")
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
    if au.battleborn:
        attacker_rules.append("Battleborn")
    if au.strider:
        attacker_rules.append("Strider")
    if au.versatile_attack:
        attacker_rules.append("Versatile Attack")
    if au.versatile_reach:
        attacker_rules.append("Versatile Reach")
    if au.unstoppable_mark:
        attacker_rules.append("Unstoppable Mark")
    if au.clan_warrior:
        attacker_rules.append("Clan Warrior" + (" Boost" if au.clan_warrior_boost else ""))
    if au.piercing_hunter:
        attacker_rules.append("Piercing Hunter")
    if au.melee_evasion:
        attacker_rules.append("Melee Evasion")
    if au.counter_attack:
        attacker_rules.append("Counter-Attack")
    if au.unpredictable_fighter:
        attacker_rules.append("Unpredictable Fighter")
    if au.bounding:
        attacker_rules.append("Bounding")
    if au.ed_teleport:
        attacker_rules.append("ED-Teleport")
    if au.vengeance:
        attacker_rules.append("Vengeance")
    if au.isr_mark:
        attacker_rules.append("Inc. Range Mark")
    if au.ignores_cover:
        attacker_rules.append("Ignores Cover")
    if au.slow:
        attacker_rules.append("Slow")
    if au.precision_fighter:
        attacker_rules.append("Precision Fighter")

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
    # Courage Aura: +1 to morale test rolls.
    courage_bonus = 1 if unit.unit.courage else 0
    passed = (_roll1() + courage_bonus) >= quality
    if not passed and unit.unit.fearless:
        passed = (_roll1() + courage_bonus) >= 4
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

    # Versatile Attack on charge — pick AP(+1) vs +1-to-hit by EV. The rule
    # gates on "enemies over 9" away" but we don't carry pre-charge distance
    # through to melee resolution; on a charge action we apply the pick
    # uniformly. Versatile Attack does not affect strike-back.
    va_ap_bonus = 0
    va_hit_bonus = 0
    if attacker.unit.versatile_attack and is_charge and not is_strike_back:
        mode = _versatile_attack_pick(attacker.unit, defender, a_quality, 0)
        if mode == 'ap':
            va_ap_bonus = 1
        else:
            va_hit_bonus = 1

    # Unstoppable Mark — when target is marked, attacks count as Unstoppable.
    # Strike-back doesn't consume the mark (the charger triggered it).
    mark_force_unstoppable = bool(defender.marked_by_unstoppable)
    if mark_force_unstoppable and not is_strike_back:
        defender.marked_by_unstoppable = False

    # ED: Melee Evasion on the defender → attackers get -1 to hit in melee.
    melee_evasion_penalty = 1 if defender.unit.melee_evasion else 0
    # ED: Precision Fighter on attacker → +1 to hit in melee.
    precision_bonus = 1 if attacker.unit.precision_fighter else 0
    # ED: Vengeance — +X to hit when target carries vengeance markers.
    vengeance_bonus = defender.vengeance_markers
    # Net hit modifier (subtracts from threshold; lower threshold = easier to hit).
    melee_hit_modifier = melee_evasion_penalty - precision_bonus - vengeance_bonus

    # ED: Unpredictable Fighter — once per engagement, roll d6: 1-3 → AP+1,
    # 4-6 → +1 hit. Applied to all weapons of this attacker for this combat.
    if attacker.unit.unpredictable_fighter:
        _ufr = _roll1()
        if _ufr <= 3:
            va_ap_bonus += 1   # piggy-back on existing va_ap_bonus channel
        else:
            va_hit_bonus += 1

    # ED: Clan Warrior army-wide hit explosion in melee.
    cw_active = attacker.unit.clan_warrior
    cw_thresh = 5 if attacker.unit.clan_warrior_boost else 6

    stat_total_attacks = 0
    stat_total_hits = 0
    stat_total_wounds = 0

    # Per-model melee weapon iteration
    melee_range_sq = MELEE_RANGE_SQ
    has_any_melee = False
    d_positions = defender.alive_positions()

    for model_idx in range(attacker.models_alive):
        if defender.models_alive <= 0:
            break

        m_col, m_row = attacker.positions[model_idx]
        model_weapons = attacker.weapons_per_model[model_idx]

        # Check if this model is in melee range of any defender model
        if not SIMPLE_MELEE:
            in_range = False
            for tc, tr in d_positions:
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
                # Thrust −1 + VA-hit −1 + ED hit modifier (melee evasion +,
                # precision fighter and vengeance markers −).
                hit_thresh = max(base_qual - (1 if thrust_active else 0)
                                 - va_hit_bonus + melee_hit_modifier, 2)

            is_nat6 = (rolls == 6)
            is_nat1 = (rolls == 1)
            is_hit = is_nat6 | ((rolls >= hit_thresh) & ~is_nat1)
            nat6_count = int(is_nat6.sum())
            normal_count = int(is_hit.sum()) - nat6_count

            # Furious extra hits on nat 6
            if furious_active:
                normal_count += nat6_count

            # Surge in melee: +1 hit per nat 6 (per-weapon).
            if weapon.surge:
                normal_count += nat6_count

            # Clan Warrior: per nat 6 (or 5+ with Boost), roll +1 attack.
            if cw_active:
                if cw_thresh == 6:
                    bonus_dice = nat6_count
                else:
                    bonus_dice = int((rolls >= cw_thresh).sum())
                if bonus_dice > 0:
                    bonus_rolls = _roll(bonus_dice)
                    b_is_nat6 = (bonus_rolls == 6)
                    b_is_nat1 = (bonus_rolls == 1)
                    b_is_hit = b_is_nat6 | ((bonus_rolls >= hit_thresh) & ~b_is_nat1)
                    b_nat6 = int(b_is_nat6.sum())
                    b_normal = int(b_is_hit.sum()) - b_nat6
                    nat6_count += b_nat6
                    normal_count += b_normal
                    stat_total_attacks += bonus_dice

            stat_total_hits += nat6_count + normal_count

            if nat6_count == 0 and normal_count == 0:
                continue

            w_ap = weapon.ap + (1 if thrust_active else 0) + va_ap_bonus
            # ED Tear / Puncture conditional AP bonuses.
            w_tear_active = _tear_active(weapon, defender)
            w_puncture_active = _puncture_active(weapon, defender)
            if w_tear_active:
                w_ap += 4
            if w_puncture_active:
                w_ap += 4
            # Fortified: reduce AP by 1 (min 0)
            if defender.unit.fortified:
                w_ap = max(w_ap - 1, 0)
            w_crack = weapon.crack
            w_rending = weapon.rending
            w_blast = weapon.blast
            w_smash_active = _smash_active(weapon, defender)
            if w_smash_active:
                w_blast = (w_blast or 0) + 3
            w_deadly = weapon.deadly
            # Bane in Melee Aura: every melee weapon picks up Bane.
            w_bane = weapon.bane or attacker.unit.bane_melee
            w_unstoppable = weapon.unstoppable or mark_force_unstoppable
            w_lacerate = weapon.lacerate
            w_shred = weapon.shred
            ignore_regen = (w_rending or w_unstoppable or w_bane
                            or w_smash_active or w_puncture_active)
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

            # --- Fast vectorized defense path (no blast/bane/lacerate/shred) ---
            if not w_blast and not w_bane and not w_lacerate and not w_shred:
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

            # --- Loop path (blast/bane/lacerate/shred) ---
            def _resolve_one_melee(def_idx: int, eff_ap: int):
                nonlocal stat_total_wounds
                if def_idx >= max_def_dice or defender.models_alive <= 0:
                    return def_idx
                _dr = int(def_rolls[def_idx]); def_idx += 1
                if w_bane and _dr == 6:
                    _dr = _roll1()
                if _dr == 6:
                    return def_idx
                block_t = d_def + eff_ap
                if block_t > 7:
                    block_t = 7
                if w_lacerate and _dr >= block_t:
                    _dr = _roll1()
                    if w_bane and _dr == 6:
                        _dr = _roll1()
                    if _dr == 6:
                        return def_idx
                if _dr >= block_t:
                    return def_idx
                stat_total_wounds += 1
                if is_deadly_weapon:
                    defender.apply_deadly_wounds(1, deadly_mult, ignore_regen)
                else:
                    defender.apply_wounds(1, ignore_regen)
                if w_shred and _dr == 1 and defender.models_alive > 0:
                    stat_total_wounds += 1
                    defender.apply_wounds(1, ignore_regen)
                return def_idx

            # Process nat6 hits
            for _ in range(nat6_count):
                if defender.models_alive <= 0:
                    break
                eff_ap = nat6_ap
                blast_n = min(w_blast, defender.models_alive) if w_blast else 1
                for _ in range(blast_n):
                    if defender.models_alive <= 0 or di >= max_def_dice:
                        break
                    di = _resolve_one_melee(di, eff_ap)

            # Process normal hits
            for _ in range(normal_count):
                if defender.models_alive <= 0:
                    break
                eff_ap = w_ap
                blast_n = min(w_blast, defender.models_alive) if w_blast else 1
                for _ in range(blast_n):
                    if defender.models_alive <= 0 or di >= max_def_dice:
                        break
                    di = _resolve_one_melee(di, eff_ap)

    # ED Vengeance trigger.
    if (defender.unit.vengeance and defender.models_alive <= 0
            and getattr(defender, "_vengeance_paid", False) is False):
        defender._vengeance_paid = True
        attacker.vengeance_markers += defender.unit.models

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

    courage_bonus = 1 if unit.unit.courage else 0
    passed = (_roll1() + courage_bonus) >= quality
    if not passed and unit.unit.fearless:
        passed = (_roll1() + courage_bonus) >= 4
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
