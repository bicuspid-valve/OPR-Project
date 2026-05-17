"""Pre-game expected ranged damage table — TERRAIN_SPEC.md §5.5.

For each (attacker_model, defender_unit) pair across the two armies, store the
expected damage U_def takes from one full activation of Y's ranged weapons,
under both cover=False and cover=True branches.

Built once after deployment & terrain freeze (and after §5.6's vis_cover table
exists, since the live cover state is read from there at game time). Stored on
the top-level game state and consumed by tactical-planning heuristics and the
ML shoot-pointer head (§5.4).

This is a planning estimate, not a dice-rolling subsystem — it should match
the policy of :func:`combat.resolve_shooting` to within reasonable closed-form
approximation. The current implementation models: quality + reliable hit
probability, AP, defense + shielded, cover (+1 to defense), deadly, blast
(capped at defender model count), and capping at total wound capacity. More
exotic abilities (Lacerate, Shred, Bane, Crack, Rending, Tear, Puncture,
Smash, Surge, Clan Warrior, etc.) are not yet folded in — extend as planning
fidelity warrants.
"""
from __future__ import annotations

from models import ResolvedUnit, UnitState, Weapon


def _starting_wounds(u: ResolvedUnit) -> int:
    return (u.tough if u.tough else 1) * u.models


def expected_damage_per_model(attacker_unit: ResolvedUnit,
                              model_weapons: list[Weapon],
                              defender_unit: ResolvedUnit,
                              cover: bool) -> float:
    """Expected wounds U_def takes from one model Y's ranged weapons in one
    activation, assuming Y is in range and has LOS to every model in U_def.

    ``cover`` adds +1 to defense (per §4.4) — suppressed per-weapon when
    ``weapon.ignores_cover`` (or unit-level ``ignores_cover``) is set (§4.5).
    """
    p_hit_default = max(0.0, (7 - attacker_unit.quality)) / 6.0
    base_def = defender_unit.defense + (1 if defender_unit.shielded else 0)
    unit_ic = bool(attacker_unit.ignores_cover)

    expected = 0.0
    for w in model_weapons:
        if w.melee:
            continue
        eff_cover = 0 if (w.ignores_cover or unit_ic) else (1 if cover else 0)
        # Cover makes saves EASIER → lowers the save threshold by 1.
        block_t = min(7, base_def + w.ap - eff_cover)
        p_block = max(0.0, (7 - block_t)) / 6.0
        p_hit = (5 / 6) if w.reliable else p_hit_default
        hits = w.attacks * p_hit
        if w.blast:
            hits *= min(w.blast, defender_unit.models)
        wounds = hits * (1.0 - p_block)
        if w.deadly:
            wounds *= w.deadly
        expected += wounds
    return min(expected, _starting_wounds(defender_unit))


def _attacker_class_key(unit: ResolvedUnit, model_weapons: list[Weapon]) -> tuple:
    """Equivalence key for attacker-side dedup.

    Per §5.5: identical models within a unit collapse to one class. Key is
    (sorted weapon name tuple, quality, ignores_cover). Other shooting-
    relevant abilities are unit-level and identical for all models in a unit
    by construction, so they don't differentiate within-unit equivalence.
    """
    weapons = tuple(sorted(w.name for w in model_weapons if not w.melee))
    return (weapons, unit.quality, bool(unit.ignores_cover))


def build_table(units_a: list[UnitState], units_b: list[UnitState]) -> dict:
    """Build the per-game expected damage table.

    Returns a dict ``table[(attacker_unit_id, model_idx, defender_unit_id)]
    -> (e_no_cover, e_cover)``. Within an attacker unit, models in the same
    equivalence class get the same numbers — computed once and replicated.
    """
    table: dict[tuple[int, int, int], tuple[float, float]] = {}

    sides: list[tuple[list[UnitState], list[UnitState]]] = [
        (units_a, units_b),
        (units_b, units_a),
    ]
    for own, opp in sides:
        for U_atk in own:
            atk_unit = U_atk.unit
            atk_id = id(U_atk)
            class_cache: dict[tuple, dict[int, tuple[float, float]]] = {}
            for mi, mw in enumerate(U_atk.weapons_per_model):
                key = _attacker_class_key(atk_unit, mw)
                if key in class_cache:
                    rows = class_cache[key]
                else:
                    rows = {}
                    for U_def in opp:
                        e_no = expected_damage_per_model(
                            atk_unit, mw, U_def.unit, cover=False)
                        e_co = expected_damage_per_model(
                            atk_unit, mw, U_def.unit, cover=True)
                        rows[id(U_def)] = (e_no, e_co)
                    class_cache[key] = rows
                for def_id, val in rows.items():
                    table[(atk_id, mi, def_id)] = val
    return table


def lookup(table: dict, attacker_unit: UnitState, model_idx: int,
           defender_unit: UnitState, cover: bool) -> float:
    """O(1) lookup. Returns 0.0 if the pair isn't in the table (e.g. dead or
    not built yet)."""
    entry = table.get((id(attacker_unit), model_idx, id(defender_unit)))
    if entry is None:
        return 0.0
    return entry[1] if cover else entry[0]
