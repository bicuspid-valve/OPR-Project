"""Evolutionary algorithm: army generation, mutation, evaluation."""
from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from models import (
    ArmyList, ArmyListEntry, ResolvedUnit, UnitState, UnitTemplate,
    compute_entry_cost, resolve_entry, merge_hero_into_unit,
)
from templates import get_templates, get_templates_dict
from game import simulate_game


# ===================================================================
# FORCE ORG
# ===================================================================

def forceorg_limits(budget: int) -> dict:
    """Derive force org limits from a points budget."""
    return {
        "max_heroes": budget // 500,
        "max_copies": 1 + (budget // 1000),
        "max_unit_cost": int(budget * 0.35),
        "max_entries": budget // 200,
    }


def _attached_hero_count(army: ArmyList) -> int:
    """Count heroes that are attached to a host unit (not standalone)."""
    return sum(1 for e in army.entries if e.attached_to >= 0)


def validate_forceorg(army: ArmyList, budget: int) -> dict:
    """Check all force org constraints. Returns dict of violations.

    Keys present only if violated:
      "heroes_over"   : int   — how many heroes over the cap
      "copies_over"   : dict[str, int] — {source_template_id: excess}
      "unit_too_expensive" : list[int] — indices of entries exceeding 35% cap
      "entries_over"  : int   — how many entries over the cap
    """
    td = get_templates_dict()
    limits = forceorg_limits(budget)
    violations: dict = {}

    attached_heroes = _attached_hero_count(army)
    effective_entries = len(army.entries) - attached_heroes
    if effective_entries > limits["max_entries"]:
        violations["entries_over"] = effective_entries - limits["max_entries"]

    # Hero count
    hero_count = sum(1 for e in army.entries if td[e.template_id].hero)
    if hero_count > limits["max_heroes"]:
        violations["heroes_over"] = hero_count - limits["max_heroes"]

    # Copy count by source_template_id (combined templates count as 2 copies)
    from collections import Counter
    copy_counts: Counter[str] = Counter()
    for e in army.entries:
        tpl = td[e.template_id]
        source = tpl.source_template_id or e.template_id
        if tpl.is_combined:
            copy_counts[source] += 2
        else:
            copy_counts[source] += 1

    copies_over: dict[str, int] = {}
    for tid, count in copy_counts.items():
        if count > limits["max_copies"]:
            copies_over[tid] = count - limits["max_copies"]
    if copies_over:
        violations["copies_over"] = copies_over

    # Single unit cost cap
    too_expensive: list[int] = []
    for i, e in enumerate(army.entries):
        if e.computed_cost > limits["max_unit_cost"]:
            too_expensive.append(i)
    if too_expensive:
        violations["unit_too_expensive"] = too_expensive

    return violations


def repair_forceorg(army: ArmyList, budget: int, enforce_forceorg: bool = True):
    """Bring army into force org compliance by removing offending entries."""
    td = get_templates_dict()

    for _ in range(20):  # safety cap
        v = validate_forceorg(army, budget)
        if not v:
            break

        # 1. Too many entries — remove cheapest
        if "entries_over" in v:
            cheapest = min(army.entries, key=lambda e: e.computed_cost)
            _remove_entry(army, army.entries.index(cheapest))
            continue

        # 2. Too many heroes — remove cheapest hero
        if "heroes_over" in v:
            heroes = [e for e in army.entries if td[e.template_id].hero]
            if heroes:
                cheapest = min(heroes, key=lambda e: e.computed_cost)
                _remove_entry(army, army.entries.index(cheapest))
            continue

        # 3. Too many copies — remove cheapest copy of over-duplicated source
        if "copies_over" in v:
            for source_tid in v["copies_over"]:
                copies = [e for e in army.entries
                          if (td[e.template_id].source_template_id or e.template_id)
                          == source_tid]
                if copies:
                    cheapest = min(copies, key=lambda e: e.computed_cost)
                    _remove_entry(army, army.entries.index(cheapest))
                    break
            continue

        # 4. Unit too expensive — strip upgrades, or replace
        if "unit_too_expensive" in v:
            idx = v["unit_too_expensive"][0]
            entry = army.entries[idx]
            limits = forceorg_limits(budget)
            # Try stripping upgrades
            if entry.chosen_upgrades:
                slot_id = random.choice(list(entry.chosen_upgrades.keys()))
                tpl = td[entry.template_id]
                opt_id = entry.chosen_upgrades[slot_id]
                # Remove dependents first
                for slot in tpl.upgrade_slots:
                    if slot.id in entry.chosen_upgrades:
                        for opt in slot.options:
                            if opt.id == entry.chosen_upgrades[slot.id] and opt.requires == opt_id:
                                del entry.chosen_upgrades[slot.id]
                                break
                del entry.chosen_upgrades[slot_id]
                compute_entry_cost(entry)
            else:
                # Replace with a random affordable unit (same faction)
                from templates import get_templates_by_faction
                templates = get_templates_by_faction(army.faction)
                affordable = [t for t in templates
                              if t.base_cost <= limits["max_unit_cost"]]
                if affordable:
                    tpl = random.choice(affordable)
                    army.entries[idx] = make_entry(tpl.id, {})
                else:
                    _remove_entry(army, idx)
            continue

    # Fill freed points (force-org-aware)
    if army.total_cost < budget - 200:
        budget_fill(army, enforce_forceorg=enforce_forceorg)

    _repair_hero_attachments(army)


def _remove_entry(army: ArmyList, idx: int):
    """Remove entry at idx and fix hero attachment indices."""
    army.entries.pop(idx)
    for e in army.entries:
        if e.attached_to == idx:
            e.attached_to = -1
        elif e.attached_to > idx:
            e.attached_to -= 1


# ===================================================================
# ARMY LIST GENERATION & MUTATION
# ===================================================================

POINTS_BUDGET = 2000

AI_ROLES = ["killer", "objective_clearer", "objective_holder", "home_objective_holder"]


def random_legal_upgrades(template: UnitTemplate) -> dict[str, str]:
    """Pick random legal upgrades for a template.

    For combined templates, tracks per-half requires separately so that
    _a slot prerequisites don't satisfy _b slot dependencies and vice versa.
    Shared (all-models) slot choices are visible to both halves.
    """
    chosen: dict[str, str] = {}

    if template.is_combined:
        shared_ids: set[str] = set()
        a_ids: set[str] = set()
        b_ids: set[str] = set()

        for slot in template.upgrade_slots:
            if random.random() < 0.5:
                if slot.id.endswith("_a"):
                    pool = shared_ids | a_ids
                elif slot.id.endswith("_b"):
                    pool = shared_ids | b_ids
                else:
                    pool = shared_ids
                valid_options = [o for o in slot.options
                                if not o.requires or o.requires in pool]
                if valid_options:
                    opt = random.choice(valid_options)
                    chosen[slot.id] = opt.id
                    if slot.id.endswith("_a"):
                        a_ids.add(opt.id)
                    elif slot.id.endswith("_b"):
                        b_ids.add(opt.id)
                    else:
                        shared_ids.add(opt.id)
    else:
        chosen_ids: set[str] = set()
        for slot in template.upgrade_slots:
            if random.random() < 0.5:
                valid_options = [o for o in slot.options
                                if not o.requires or o.requires in chosen_ids]
                if valid_options:
                    opt = random.choice(valid_options)
                    chosen[slot.id] = opt.id
                    chosen_ids.add(opt.id)

    return chosen


def _default_combat_preference(tpl: UnitTemplate) -> str:
    """Assign combat_preference based on weapon profile."""
    has_ranged_long = any(w.range_inches > 12 and not w.melee for w in tpl.base_weapons)
    has_melee = any(w.melee for w in tpl.base_weapons)

    if has_melee and not has_ranged_long:
        # Melee-focused or pistol-only: 70% melee
        return "melee" if random.random() < 0.7 else "ranged"
    elif has_ranged_long:
        # Has real ranged weapons: 70% ranged
        return "ranged" if random.random() < 0.7 else "melee"
    else:
        return "ranged"


def make_entry(template_id: str, upgrades: dict[str, str] | None = None,
               ai_role: str | None = None) -> ArmyListEntry:
    tpl = get_templates_dict()[template_id]
    if upgrades is None:
        upgrades = random_legal_upgrades(tpl)
    if ai_role is None:
        ai_role = random.choice(AI_ROLES)
    combat_pref = _default_combat_preference(tpl)
    entry = ArmyListEntry(template_id, upgrades, ai_role=ai_role,
                          combat_preference=combat_pref)
    compute_entry_cost(entry)
    return entry


def _ensure_holder(army: ArmyList):
    """Ensure at least one objective-holder exists and combat preferences are balanced."""
    has_holder = any(e.ai_role == "objective_holder" for e in army.entries)
    if not has_holder and army.entries:
        cheapest = min(army.entries, key=lambda e: e.computed_cost)
        cheapest.ai_role = "objective_holder"

    # Ensure balanced combat preferences
    templates = get_templates_dict()
    has_melee_pref = any(e.combat_preference == "melee" for e in army.entries)
    has_ranged_pref = any(e.combat_preference == "ranged" for e in army.entries)

    if not has_melee_pref and army.entries:
        # Find a melee-capable unit and set it to melee preference
        melee_capable = [e for e in army.entries
                         if any(w.melee for w in templates[e.template_id].base_weapons)]
        if melee_capable:
            random.choice(melee_capable).combat_preference = "melee"

    if not has_ranged_pref and army.entries:
        ranged_capable = [e for e in army.entries
                          if any(w.range_inches > 12 and not w.melee
                                 for w in templates[e.template_id].base_weapons)]
        if ranged_capable:
            random.choice(ranged_capable).combat_preference = "ranged"


def _attach_heroes(army: ArmyList):
    """Randomly attach hero entries to non-hero host units (70% chance)."""
    templates = get_templates_dict()
    hosted: set[int] = set()  # indices already hosting a hero

    for i, entry in enumerate(army.entries):
        tpl = templates.get(entry.template_id)
        if not tpl or not tpl.hero:
            continue
        if random.random() < 0.3:
            entry.attached_to = -1  # 30% unattached
            continue
        # Find valid host units (non-hero, multi-model, not already hosting)
        valid_hosts = [j for j, e in enumerate(army.entries)
                       if j != i and not templates[e.template_id].hero
                       and templates[e.template_id].size > 1
                       and j not in hosted]
        if valid_hosts:
            host_idx = random.choice(valid_hosts)
            entry.attached_to = host_idx
            entry.ai_role = army.entries[host_idx].ai_role
            hosted.add(host_idx)
        else:
            entry.attached_to = -1


def _repair_hero_attachments(army: ArmyList):
    """Validate and fix invalid hero attachments after mutations."""
    templates = get_templates_dict()
    hosted: set[int] = set()

    for i, entry in enumerate(army.entries):
        tpl = templates.get(entry.template_id)
        if not tpl or not tpl.hero:
            entry.attached_to = -1
            continue
        if entry.attached_to < 0:
            continue
        # Validate target
        valid = True
        if entry.attached_to >= len(army.entries):
            valid = False
        elif entry.attached_to == i:
            valid = False
        elif templates[army.entries[entry.attached_to].template_id].hero:
            valid = False
        elif templates[army.entries[entry.attached_to].template_id].size <= 1:
            valid = False
        elif entry.attached_to in hosted:
            valid = False

        if valid:
            hosted.add(entry.attached_to)
            entry.ai_role = army.entries[entry.attached_to].ai_role
        else:
            entry.attached_to = -1


def _forceorg_filter(templates_list: list[UnitTemplate], army: ArmyList,
                     budget: int) -> list[UnitTemplate]:
    """Filter templates to those that won't violate force org if added."""
    td = get_templates_dict()
    limits = forceorg_limits(budget)

    attached_heroes = _attached_hero_count(army)
    effective_entries = len(army.entries) - attached_heroes
    if effective_entries >= limits["max_entries"]:
        return []

    hero_count = sum(1 for e in army.entries if td[e.template_id].hero)

    from collections import Counter
    copy_counts: Counter[str] = Counter()
    for e in army.entries:
        etpl = td[e.template_id]
        source = etpl.source_template_id or e.template_id
        if etpl.is_combined:
            copy_counts[source] += 2
        else:
            copy_counts[source] += 1

    result = []
    for tpl in templates_list:
        if tpl.base_cost > limits["max_unit_cost"]:
            continue
        if tpl.hero and hero_count >= limits["max_heroes"]:
            continue
        # Check copy count: combined adds 2, normal adds 1
        source = tpl.source_template_id or tpl.id
        add_count = 2 if tpl.is_combined else 1
        if copy_counts[source] + add_count > limits["max_copies"]:
            continue
        result.append(tpl)
    return result


def generate_random_army(mode: str = "objectives",
                         enforce_forceorg: bool = False,
                         faction: str = "hef") -> ArmyList:
    """Generate a random valid army list for the given faction.

    *faction* is one of "hef" / "bb" — armies are homogeneous (no mixing).
    """
    from templates import get_templates_by_faction
    templates = get_templates_by_faction(faction)
    army = ArmyList(faction=faction)
    remaining = POINTS_BUDGET
    fails = 0

    while remaining > 0 and fails < 10:
        if enforce_forceorg:
            pool = _forceorg_filter(templates, army, POINTS_BUDGET)
            pool = [t for t in pool if t.base_cost <= remaining]
            if not pool:
                break
            tpl = random.choice(pool)
        else:
            tpl = random.choice(templates)
        entry = make_entry(tpl.id)
        if entry.computed_cost <= remaining:
            if enforce_forceorg and entry.computed_cost > forceorg_limits(POINTS_BUDGET)["max_unit_cost"]:
                fails += 1
                continue
            army.entries.append(entry)
            remaining -= entry.computed_cost
            fails = 0
        else:
            fails += 1

    if army.total_cost < 1500:
        return generate_random_army(mode, enforce_forceorg=enforce_forceorg,
                                    faction=faction)

    _attach_heroes(army)
    if mode != "kill_points":
        _ensure_holder(army)
    return army


def budget_fill(army: ArmyList, enforce_forceorg: bool = False):
    """Try to fill remaining budget with cheap units."""
    from templates import get_templates_by_faction
    remaining = POINTS_BUDGET - army.total_cost
    templates = get_templates_by_faction(army.faction)
    for _ in range(5):
        if remaining < 80:  # min viable unit cost ~90
            break
        if enforce_forceorg:
            pool = _forceorg_filter(templates, army, POINTS_BUDGET)
            affordable = [t for t in pool if t.base_cost <= remaining]
        else:
            affordable = [t for t in templates if t.base_cost <= remaining]
        if not affordable:
            break
        tpl = random.choice(affordable)
        entry = make_entry(tpl.id, {})  # no upgrades to keep it cheap
        compute_entry_cost(entry)
        if entry.computed_cost <= remaining:
            army.entries.append(entry)
            remaining -= entry.computed_cost


def repair_budget(army: ArmyList, enforce_forceorg: bool = False):
    """Bring army under budget by removing upgrades then units."""
    while army.total_cost > POINTS_BUDGET and army.entries:
        # Try removing a random upgrade first
        entries_with_upgrades = [e for e in army.entries if e.chosen_upgrades]
        if entries_with_upgrades:
            entry = random.choice(entries_with_upgrades)
            slot_id = random.choice(list(entry.chosen_upgrades.keys()))
            # Check if anything depends on this upgrade
            opt_id = entry.chosen_upgrades[slot_id]
            tpl = get_templates_dict()[entry.template_id]
            # Remove dependents first
            for slot in tpl.upgrade_slots:
                if slot.id in entry.chosen_upgrades:
                    for opt in slot.options:
                        if opt.id == entry.chosen_upgrades[slot.id] and opt.requires == opt_id:
                            del entry.chosen_upgrades[slot.id]
                            break
            del entry.chosen_upgrades[slot_id]
            compute_entry_cost(entry)
        else:
            # Remove cheapest unit
            cheapest = min(army.entries, key=lambda e: e.computed_cost)
            _remove_entry(army, army.entries.index(cheapest))

    if army.total_cost < POINTS_BUDGET - 200:
        budget_fill(army, enforce_forceorg=enforce_forceorg)


MUTATION_WEIGHTS = {
    'add_unit': 20,
    'remove_unit': 15,
    'replace_unit': 25,
    'toggle_upgrade': 25,
    'swap_upgrade': 15,
    'change_role': 20,
    'swap_roles': 10,
    'change_combat_preference': 15,
    'attach_hero': 10,
    'detach_hero': 10,
    'move_hero': 10,
}


def mutate(army: ArmyList, mode: str = "objectives",
           enforce_forceorg: bool = False):
    """Apply one random mutation to the army list."""
    operators = list(MUTATION_WEIGHTS.keys())
    weights = list(MUTATION_WEIGHTS.values())
    # Combined template mutations (always available)
    operators.extend(['merge_to_combined_template', 'split_combined_template'])
    weights.extend([8, 8])
    if mode == "kill_points":
        # Skip role mutations — roles are unused in kill points mode
        for skip in ('change_role', 'swap_roles'):
            idx = operators.index(skip)
            operators.pop(idx)
            weights.pop(idx)
    op = random.choices(operators, weights=weights, k=1)[0]
    from templates import get_templates_by_faction
    templates = get_templates_by_faction(army.faction)
    td = get_templates_dict()

    if op == 'add_unit':
        if enforce_forceorg:
            pool = _forceorg_filter(templates, army, POINTS_BUDGET)
            if not pool:
                pass  # no-op
            else:
                tpl = random.choice(pool)
                entry = make_entry(tpl.id)
                if enforce_forceorg and entry.computed_cost > forceorg_limits(POINTS_BUDGET)["max_unit_cost"]:
                    pass  # skip
                else:
                    army.entries.append(entry)
        else:
            tpl = random.choice(templates)
            entry = make_entry(tpl.id)
            army.entries.append(entry)

    elif op == 'remove_unit':
        if len(army.entries) > 1:
            idx = random.randrange(len(army.entries))
            _remove_entry(army, idx)
            budget_fill(army, enforce_forceorg=enforce_forceorg)

    elif op == 'replace_unit':
        if army.entries:
            idx = random.randrange(len(army.entries))
            old_cost = army.entries[idx].computed_cost
            candidates = [t for t in templates if abs(t.base_cost - old_cost) < 150]
            if not candidates:
                candidates = templates
            if enforce_forceorg:
                # Temporarily remove entry to get accurate filter
                saved = army.entries.pop(idx)
                pool = _forceorg_filter(candidates, army, POINTS_BUDGET)
                army.entries.insert(idx, saved)
                if pool:
                    candidates = pool
            tpl = random.choice(candidates)
            new_entry = make_entry(tpl.id)
            army.entries[idx] = new_entry

    elif op == 'toggle_upgrade':
        if army.entries:
            idx = random.randrange(len(army.entries))
            entry = army.entries[idx]
            tpl = td[entry.template_id]
            if tpl.upgrade_slots:
                slot = random.choice(tpl.upgrade_slots)
                old_upgrades = dict(entry.chosen_upgrades)
                if slot.id in entry.chosen_upgrades:
                    # Also remove dependents
                    removed_opt_id = entry.chosen_upgrades[slot.id]
                    del entry.chosen_upgrades[slot.id]
                    for s2 in tpl.upgrade_slots:
                        if s2.id in entry.chosen_upgrades:
                            for o in s2.options:
                                if o.id == entry.chosen_upgrades[s2.id] and o.requires == removed_opt_id:
                                    del entry.chosen_upgrades[s2.id]
                                    break
                else:
                    chosen_ids = set(entry.chosen_upgrades.values())
                    valid = [o for o in slot.options
                             if not o.requires or o.requires in chosen_ids]
                    if valid:
                        entry.chosen_upgrades[slot.id] = random.choice(valid).id
                compute_entry_cost(entry)
                # Revert if over 35% cap
                if enforce_forceorg:
                    max_cost = forceorg_limits(POINTS_BUDGET)["max_unit_cost"]
                    if entry.computed_cost > max_cost:
                        entry.chosen_upgrades = old_upgrades
                        compute_entry_cost(entry)

    elif op == 'swap_upgrade':
        if army.entries:
            idx = random.randrange(len(army.entries))
            entry = army.entries[idx]
            tpl = td[entry.template_id]
            used = [s for s in tpl.upgrade_slots if s.id in entry.chosen_upgrades]
            if used:
                slot = random.choice(used)
                current = entry.chosen_upgrades[slot.id]
                old_cost = entry.computed_cost
                chosen_ids = set(entry.chosen_upgrades.values())
                others = [o for o in slot.options
                          if o.id != current and (not o.requires or o.requires in chosen_ids)]
                if others:
                    entry.chosen_upgrades[slot.id] = random.choice(others).id
                    compute_entry_cost(entry)
                    # Revert if over 35% cap
                    if enforce_forceorg:
                        max_cost = forceorg_limits(POINTS_BUDGET)["max_unit_cost"]
                        if entry.computed_cost > max_cost:
                            entry.chosen_upgrades[slot.id] = current
                            compute_entry_cost(entry)

    elif op == 'merge_to_combined_template':
        # Find two entries with same non-combined source template and matching
        # all-models upgrades, replace with a single combined template entry
        from collections import defaultdict
        by_source: dict[str, list[int]] = defaultdict(list)
        for i, e in enumerate(army.entries):
            tpl_e = td[e.template_id]
            if not tpl_e.is_combined and not tpl_e.hero and tpl_e.size > 1:
                by_source[e.template_id].append(i)
        mergeable = [(sid, idxs) for sid, idxs in by_source.items()
                     if len(idxs) >= 2]
        if mergeable:
            source_id, idxs = random.choice(mergeable)
            i, j = random.sample(idxs, 2)
            combined_tid = f"{source_id}_combined"
            if combined_tid in td:
                source_tpl = td[source_id]
                # Check all-models upgrade match
                match = True
                for slot in source_tpl.upgrade_slots:
                    if slot.options and all(o.applies_to_all for o in slot.options):
                        a_val = army.entries[i].chosen_upgrades.get(slot.id)
                        b_val = army.entries[j].chosen_upgrades.get(slot.id)
                        if a_val != b_val:
                            match = False
                            break
                if match:
                    # Build combined upgrades
                    combined_upgrades: dict[str, str] = {}
                    for slot in source_tpl.upgrade_slots:
                        is_shared = slot.options and all(
                            o.applies_to_all for o in slot.options)
                        if is_shared:
                            val = army.entries[i].chosen_upgrades.get(slot.id)
                            if val:
                                combined_upgrades[slot.id] = val
                        else:
                            a_val = army.entries[i].chosen_upgrades.get(slot.id)
                            b_val = army.entries[j].chosen_upgrades.get(slot.id)
                            if a_val:
                                combined_upgrades[f"{slot.id}_a"] = a_val
                            if b_val:
                                combined_upgrades[f"{slot.id}_b"] = b_val
                    role = army.entries[i].ai_role
                    pref = army.entries[i].combat_preference
                    # Remove both entries (higher index first)
                    hi, lo = max(i, j), min(i, j)
                    _remove_entry(army, hi)
                    _remove_entry(army, lo)
                    new_entry = ArmyListEntry(
                        combined_tid, combined_upgrades,
                        ai_role=role, combat_preference=pref)
                    compute_entry_cost(new_entry)
                    army.entries.append(new_entry)

    elif op == 'split_combined_template':
        # Find a combined template entry and split into two normal entries
        combined_entries = [(i, e) for i, e in enumerate(army.entries)
                           if td[e.template_id].is_combined]
        if combined_entries:
            idx, entry = random.choice(combined_entries)
            comb_tpl = td[entry.template_id]
            source_id = comb_tpl.source_template_id
            if source_id in td:
                source_tpl = td[source_id]
                # Distribute upgrades to two entries
                upgrades_a: dict[str, str] = {}
                upgrades_b: dict[str, str] = {}
                for slot_id, opt_id in entry.chosen_upgrades.items():
                    if slot_id.endswith("_a"):
                        upgrades_a[slot_id[:-2]] = opt_id
                    elif slot_id.endswith("_b"):
                        upgrades_b[slot_id[:-2]] = opt_id
                    else:
                        # Shared slot — both halves get it
                        upgrades_a[slot_id] = opt_id
                        upgrades_b[slot_id] = opt_id
                role = entry.ai_role
                pref = entry.combat_preference
                _remove_entry(army, idx)
                e_a = ArmyListEntry(source_id, upgrades_a,
                                    ai_role=role, combat_preference=pref)
                e_b = ArmyListEntry(source_id, upgrades_b,
                                    ai_role=role, combat_preference=pref)
                compute_entry_cost(e_a)
                compute_entry_cost(e_b)
                army.entries.extend([e_a, e_b])

    elif op == 'change_role':
        if army.entries:
            idx = random.randrange(len(army.entries))
            entry = army.entries[idx]
            role_cycle = {"killer": "objective_clearer",
                          "objective_clearer": "objective_holder",
                          "objective_holder": "home_objective_holder",
                          "home_objective_holder": "killer"}
            entry.ai_role = role_cycle.get(entry.ai_role, "killer")
            # Propagate to any attached hero
            for e in army.entries:
                if e.attached_to == idx:
                    e.ai_role = entry.ai_role

    elif op == 'swap_roles':
        if len(army.entries) >= 2:
            i, j = random.sample(range(len(army.entries)), 2)
            if army.entries[i].ai_role != army.entries[j].ai_role:
                army.entries[i].ai_role, army.entries[j].ai_role = \
                    army.entries[j].ai_role, army.entries[i].ai_role
                # Propagate to any attached heroes
                for e in army.entries:
                    if e.attached_to == i:
                        e.ai_role = army.entries[i].ai_role
                    elif e.attached_to == j:
                        e.ai_role = army.entries[j].ai_role

    elif op == 'change_combat_preference':
        if army.entries:
            idx = random.randrange(len(army.entries))
            entry = army.entries[idx]
            entry.combat_preference = "melee" if entry.combat_preference == "ranged" else "ranged"

    elif op == 'attach_hero':
        td = get_templates_dict()
        unattached_heroes = [(i, e) for i, e in enumerate(army.entries)
                             if td[e.template_id].hero and e.attached_to < 0]
        if unattached_heroes:
            _, hero_entry = random.choice(unattached_heroes)
            hosted = {e.attached_to for e in army.entries if e.attached_to >= 0}
            valid_hosts = [j for j, e in enumerate(army.entries)
                           if not td[e.template_id].hero
                           and td[e.template_id].size > 1
                           and j not in hosted]
            if valid_hosts:
                host_idx = random.choice(valid_hosts)
                hero_entry.attached_to = host_idx
                hero_entry.ai_role = army.entries[host_idx].ai_role

    elif op == 'detach_hero':
        td = get_templates_dict()
        attached_heroes = [(i, e) for i, e in enumerate(army.entries)
                           if td[e.template_id].hero and e.attached_to >= 0]
        if attached_heroes:
            _, hero_entry = random.choice(attached_heroes)
            hero_entry.attached_to = -1

    elif op == 'move_hero':
        td = get_templates_dict()
        attached_heroes = [(i, e) for i, e in enumerate(army.entries)
                           if td[e.template_id].hero and e.attached_to >= 0]
        if attached_heroes:
            _, hero_entry = random.choice(attached_heroes)
            hosted = {e.attached_to for e in army.entries
                      if e.attached_to >= 0 and e is not hero_entry}
            valid_hosts = [j for j, e in enumerate(army.entries)
                           if not td[e.template_id].hero
                           and td[e.template_id].size > 1
                           and j not in hosted
                           and j != hero_entry.attached_to]
            if valid_hosts:
                host_idx = random.choice(valid_hosts)
                hero_entry.attached_to = host_idx
                hero_entry.ai_role = army.entries[host_idx].ai_role

    _repair_hero_attachments(army)

    if army.total_cost > POINTS_BUDGET:
        repair_budget(army, enforce_forceorg=enforce_forceorg)

    if enforce_forceorg:
        repair_forceorg(army, POINTS_BUDGET, enforce_forceorg=enforce_forceorg)

    if not army.entries:
        seed_pool = (_forceorg_filter(templates, army, POINTS_BUDGET)
                     if enforce_forceorg else templates)
        if not seed_pool:
            seed_pool = [t for t in templates
                         if t.base_cost <= forceorg_limits(POINTS_BUDGET)["max_unit_cost"]]
        army.entries = [make_entry(random.choice(seed_pool).id)]
        budget_fill(army, enforce_forceorg=enforce_forceorg)

    if mode != "kill_points":
        _ensure_holder(army)


# ===================================================================
# DIVERSITY & HALL OF FAME CONSTANTS
# ===================================================================

SIMILARITY_THRESHOLD = 0.5   # armies above this are "very similar"
DIVERSITY_DECAY_RATE = 0.8   # adjusted fitness multiplier per similar higher-ranked army
HOF_MAX_SIZE = 30             # max Hall of Fame members (global)
HOF_PER_FACTION_MAX_SIZE = 15 # max per-faction Hall of Fame members
HOF_PER_FACTION_CANDIDATE_COUNT = 3  # top N per faction considered each cycle
HOF_EVAL_INTERVAL = 10        # generations between HoF evaluations
HOF_CANDIDATE_COUNT = 5       # population members considered for HoF promotion
HOF_GAMES_PER_MATCHUP = 10    # games per pairing in HoF round-robin


# ===================================================================
# SIMILARITY METRIC
# ===================================================================

UNIT_BASE_SIMILARITY = 0.3   # minimum similarity for same-template units

def unit_similarity(a: ArmyListEntry, b: ArmyListEntry) -> float:
    """Similarity for two units: 0.0 if different template, otherwise
    BASE + (1-BASE)*jaccard so same-template always scores >= 0.3."""
    if a.template_id != b.template_id:
        return 0.0
    set_a = set(a.chosen_upgrades.items())
    set_b = set(b.chosen_upgrades.items())
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    jaccard = len(set_a & set_b) / len(union)
    return UNIT_BASE_SIMILARITY + (1.0 - UNIT_BASE_SIMILARITY) * jaccard


def army_composition_match(army_a: ArmyList, army_b: ArmyList) -> bool:
    """True if two armies have identical units and upgrades (ignoring AI assignments)."""
    return army_similarity(army_a, army_b) >= 1.0


def army_identity_match(army_a: ArmyList, army_b: ArmyList) -> bool:
    """True if two armies are fully identical including AI role assignments."""
    if not army_composition_match(army_a, army_b):
        return False
    # Same composition — now check AI assignments match exactly.
    # Sort entries by (template_id, sorted upgrades) for stable comparison.
    def _sort_key(e: ArmyListEntry):
        return (e.template_id, sorted(e.chosen_upgrades.items()))
    sorted_a = sorted(army_a.entries, key=_sort_key)
    sorted_b = sorted(army_b.entries, key=_sort_key)
    return all(
        a.ai_role == b.ai_role and a.combat_preference == b.combat_preference
        for a, b in zip(sorted_a, sorted_b)
    )


def army_similarity(army_a: ArmyList, army_b: ArmyList) -> float:
    """Greedy bipartite matching on unit similarities, weighted by unit cost.

    Each matched pair contributes  similarity * avg_cost_of_pair,
    normalised by the larger army's total cost so the result stays in [0, 1].
    Expensive units therefore count proportionally more than cheap ones.
    """
    entries_a = army_a.entries
    entries_b = army_b.entries
    if not entries_a or not entries_b:
        return 0.0

    na, nb = len(entries_a), len(entries_b)
    costs_a = [e.computed_cost for e in entries_a]
    costs_b = [e.computed_cost for e in entries_b]

    # Build pairwise similarity matrix
    sim = [[unit_similarity(entries_a[i], entries_b[j])
            for j in range(nb)] for i in range(na)]

    # Greedy matching: repeatedly pick highest unmatched similarity
    matched_a: set[int] = set()
    matched_b: set[int] = set()
    total = 0.0
    pairs_possible = min(na, nb)

    # Collect all (sim, i, j) and sort descending
    candidates = []
    for i in range(na):
        for j in range(nb):
            if sim[i][j] > 0:
                candidates.append((sim[i][j], i, j))
    candidates.sort(reverse=True)

    for s, i, j in candidates:
        if i in matched_a or j in matched_b:
            continue
        avg_cost = (costs_a[i] + costs_b[j]) / 2.0
        total += s * avg_cost
        matched_a.add(i)
        matched_b.add(j)
        if len(matched_a) == pairs_possible:
            break

    norm = max(sum(costs_a), sum(costs_b))
    return total / norm if norm > 0 else 0.0


def _compute_adjusted_fitness(population: list[ArmyList]) -> list[float]:
    """Compute diversity-penalised fitness for selection.

    Sorts by raw fitness descending, then for each army counts how many
    higher-ranked armies exceed SIMILARITY_THRESHOLD. Adjusted fitness
    decays exponentially per similar-and-higher army.
    """
    n = len(population)
    indexed = sorted(range(n), key=lambda i: population[i].fitness, reverse=True)

    # Build pairwise similarity matrix (only upper triangle needed)
    sim_cache: dict[tuple[int, int], float] = {}
    for idx_a in range(n):
        for idx_b in range(idx_a + 1, n):
            i, j = indexed[idx_a], indexed[idx_b]
            key = (min(i, j), max(i, j))
            if key not in sim_cache:
                sim_cache[key] = army_similarity(population[i], population[j])

    adjusted = [0.0] * n
    for rank, pop_idx in enumerate(indexed):
        # Count how many higher-ranked armies are similar
        count = 0
        for higher_rank in range(rank):
            higher_idx = indexed[higher_rank]
            key = (min(pop_idx, higher_idx), max(pop_idx, higher_idx))
            if sim_cache.get(key, 0.0) >= SIMILARITY_THRESHOLD:
                count += 1
        adjusted[pop_idx] = population[pop_idx].fitness * (DIVERSITY_DECAY_RATE ** count)

    return adjusted


# ===================================================================
# HALL OF FAME
# ===================================================================

@dataclass
class HallOfFameEntry:
    army: ArmyList
    fitness: float
    generation_added: int


class HallOfFame:
    """Curated archive of all-time best armies.

    *max_size* — keep at most this many entries.
    *faction_filter* — when non-empty, only armies of this faction are eligible
    for promotion AND only same-faction matchups happen in the round-robin.
    """

    def __init__(self, max_size: int = HOF_MAX_SIZE, faction_filter: str = ""):
        self.entries: list[HallOfFameEntry] = []
        self.max_size = max_size
        self.faction_filter = faction_filter

    @classmethod
    def load_from_json(cls, path: str | Path,
                       enforce_forceorg: bool = False) -> "HallOfFame":
        """Load a previously saved Hall of Fame from a JSON file.

        When *enforce_forceorg* is True, every loaded army is validated
        against forceorg_limits(POINTS_BUDGET); non-compliant armies are
        repaired in place. Repaired armies keep their saved fitness as a
        prior; the next round-robin in try_promote will overwrite it.
        """
        import json
        hof = cls()
        p = Path(path)
        if not p.exists():
            return hof
        with open(p) as f:
            data = json.load(f)
        repaired = 0
        for item in data:
            entries = []
            for e in item['entries']:
                entries.append(ArmyListEntry(
                    template_id=e['template_id'],
                    chosen_upgrades=e['upgrades'],
                    ai_role=e['ai_role'],
                    combat_preference=e['combat_preference'],
                    computed_cost=e['cost'],
                    attached_to=e.get('attached_to', -1),
                ))
            army = ArmyList(
                entries=entries,
                fitness=item['fitness'],
                faction=item.get('faction', 'hef'),
            )
            if enforce_forceorg and validate_forceorg(army, POINTS_BUDGET):
                repair_forceorg(army, POINTS_BUDGET, enforce_forceorg=True)
                repaired += 1
            hof.entries.append(HallOfFameEntry(
                army=army,
                fitness=item['fitness'],
                generation_added=item['generation_added'],
            ))
        if repaired:
            print(f"  [HoF load] {repaired} army(s) from {p.name} "
                  f"repaired to satisfy force-org.")
        return hof

    def try_promote(self, population: list[ArmyList], mode: str,
                    pool: ProcessPoolExecutor, generation: int,
                    ml_coroutine_batch: bool = False) -> dict:
        """Evaluate top candidates against current HoF members.

        Returns a dict with keys: candidates, promoted, demoted, size, top_fitness.
        """
        # 1. Select top candidates by raw fitness, excluding full duplicates
        #    (same units, upgrades, AND AI assignments) of existing HoF members.
        #    For a faction-scoped HoF, only same-faction armies are eligible.
        eligible = (population if not self.faction_filter
                    else [a for a in population if a.faction == self.faction_filter])
        # Per-faction HoFs use a smaller candidate slice (top 3 vs default 5).
        candidate_limit = (HOF_CANDIDATE_COUNT if not self.faction_filter
                           else HOF_PER_FACTION_CANDIDATE_COUNT)
        ranked = sorted(eligible, key=lambda a: a.fitness, reverse=True)
        candidates: list[ArmyList] = []
        for a in ranked:
            if len(candidates) >= candidate_limit:
                break
            is_dup = any(army_identity_match(a, e.army) for e in self.entries)
            if not is_dup:
                candidates.append(copy.deepcopy(a))

        # 2. Build evaluation pool: candidates + current HoF members
        eval_armies: list[ArmyList] = []
        eval_armies.extend(candidates)
        eval_armies.extend(e.army for e in self.entries)

        n = len(eval_armies)
        if n < 2:
            # Not enough armies to run a round-robin — just add candidates
            for c in candidates:
                if len(self.entries) < self.max_size:
                    self.entries.append(HallOfFameEntry(c, c.fitness, generation))
                    # (duplicates already filtered above)
            top_fit = max((e.fitness for e in self.entries), default=0.0)
            return {
                'candidates': len(candidates), 'promoted': len(candidates),
                'demoted': 0, 'size': len(self.entries), 'top_fitness': top_fit,
            }

        # 3. Round-robin tournament
        resolved = [resolve_army(a) for a in eval_armies]
        wins = [0.0] * n
        games = [0] * n

        matchup_indices = []
        for i in range(n):
            for j in range(i + 1, n):
                matchup_indices.append((i, j))

        if ml_coroutine_batch:
            # Chunk matchups across workers for batched cross-game inference
            n_workers = _WORKER_COUNT
            chunks = [[] for _ in range(n_workers)]
            pair_chunk_map = []  # (chunk_idx, position_in_chunk)
            for p_idx, (i, j) in enumerate(matchup_indices):
                c_idx = p_idx % n_workers
                chunks[c_idx].append((eval_armies[i], eval_armies[j],
                                      resolved[i], resolved[j]))
                pair_chunk_map.append((c_idx, len(chunks[c_idx]) - 1))

            chunk_work = [(chunk, mode, HOF_GAMES_PER_MATCHUP)
                          for chunk in chunks if chunk]
            chunk_results = list(pool.map(_play_matchup_batched, chunk_work))

            for p_idx, (i, j) in enumerate(matchup_indices):
                c_idx, pos = pair_chunk_map[p_idx]
                a_w, b_w = chunk_results[c_idx][pos]
                wins[i] += a_w
                wins[j] += b_w
                games[i] += HOF_GAMES_PER_MATCHUP
                games[j] += HOF_GAMES_PER_MATCHUP
        else:
            work = [(eval_armies[i], eval_armies[j],
                     resolved[i], resolved[j],
                     mode, HOF_GAMES_PER_MATCHUP)
                    for i, j in matchup_indices]
            results = list(pool.map(_play_matchup, work, chunksize=10))

            for (i, j), (a_w, b_w) in zip(matchup_indices, results):
                wins[i] += a_w
                wins[j] += b_w
                games[i] += HOF_GAMES_PER_MATCHUP
                games[j] += HOF_GAMES_PER_MATCHUP

        win_rates = [wins[i] / max(games[i], 1) for i in range(n)]

        # 4. Rank by win rate, keep top self.max_size
        num_candidates = len(candidates)
        old_size = len(self.entries)
        pool_entries: list[HallOfFameEntry] = []

        for i in range(n):
            if i < num_candidates:
                # Sync the army's own fitness to the HoF round-robin result
                # so format_army's "Win rate" matches the HoF ranking.
                eval_armies[i].fitness = win_rates[i]
                pool_entries.append(HallOfFameEntry(
                    eval_armies[i], win_rates[i], generation))
            else:
                hof_idx = i - num_candidates
                entry = self.entries[hof_idx]
                entry.fitness = win_rates[i]
                entry.army.fitness = win_rates[i]
                pool_entries.append(entry)

        pool_entries.sort(key=lambda e: e.fitness, reverse=True)

        # Demote lower-scoring composition twins (same units/upgrades,
        # different AI assignments) to the bottom 5 slots.
        bottom_start = max(0, self.max_size - 5)
        claimed: set[int] = set()  # indices of "winner" entries
        demote_indices: list[int] = []
        for i, entry_i in enumerate(pool_entries):
            if i in claimed:
                continue
            for j in range(i + 1, len(pool_entries)):
                if j in claimed or j in set(demote_indices):
                    continue
                if army_composition_match(entry_i.army, pool_entries[j].army):
                    # i is higher-ranked (better fitness) — demote j
                    demote_indices.append(j)
            claimed.add(i)
        if demote_indices:
            demoted_entries = [pool_entries[j] for j in demote_indices]
            kept = [e for i, e in enumerate(pool_entries) if i not in set(demote_indices)]
            # Insert demoted entries at the bottom-5 boundary
            insert_pos = min(bottom_start, len(kept))
            pool_entries = kept[:insert_pos] + demoted_entries + kept[insert_pos:]

        self.entries = pool_entries[:self.max_size]

        # Count promotions/demotions
        new_armies = {id(c) for c in candidates}
        promoted = sum(1 for e in self.entries if id(e.army) in new_armies)
        demoted = max(0, old_size - (len(self.entries) - promoted))
        top_fit = self.entries[0].fitness if self.entries else 0.0

        return {
            'candidates': num_candidates, 'promoted': promoted,
            'demoted': demoted, 'size': len(self.entries),
            'top_fitness': top_fit,
        }

    def format_report(self, mode: str = "objectives",
                      enforce_forceorg: bool = False) -> str:
        """Format a printable Hall of Fame report."""
        if not self.entries:
            return "=== HALL OF FAME === (empty)"
        lines = [f"=== HALL OF FAME (top {len(self.entries)} all-time) ==="]
        # Import here to avoid circular dependency
        from main import format_army
        for i, entry in enumerate(self.entries):
            lines.append(f"#{i+1}  [{entry.fitness:.3f}] (gen {entry.generation_added})")
            lines.append(format_army(entry.army, mode=mode,
                                     enforce_forceorg=enforce_forceorg))
            lines.append("")
        return "\n".join(lines)


# ===================================================================
# EVOLUTIONARY ALGORITHM
# ===================================================================

POPULATION_SIZE = 150
META_CHASERS = 75                  # split 25/25/25 across factions at init
HARDCORE_FANS_PER_FACTION = 25     # × 3 factions = 75
ELITE_META = 15                    # top meta-chasers preserved each gen
ELITE_PER_FAN = 5                  # top hardcore fans per faction preserved each gen
META_OFFSPRING = META_CHASERS - ELITE_META                       # 60
FAN_OFFSPRING_PER_FACTION = HARDCORE_FANS_PER_FACTION - ELITE_PER_FAN  # 20
ELITE_COUNT = ELITE_META + 3 * ELITE_PER_FAN                     # 30
OFFSPRING_COUNT = POPULATION_SIZE - ELITE_COUNT                  # 120
FACTIONS = ("hef", "bb", "ed")
GAMES_PER_MATCHUP = 2 #3
GENERATIONS = 100000 #200
TOURNAMENT_SIZE = 4
SWISS_ROUNDS = 10  # Swiss-system rounds (ceil(log2(N)) ≈ 7, +2 for ranking stability)
TIME_LIMIT = 540  # wall-clock limit in minutes; None = no limit


def resolve_army(army: ArmyList) -> list[ResolvedUnit]:
    """Resolve all entries into ResolvedUnits.
    Combined templates resolve directly to doubled units via resolve_entry."""
    return [resolve_entry(entry) for entry in army.entries]


def _make_unit_states(army: ArmyList, resolved: list[ResolvedUnit],
                      player: str) -> list[UnitState]:
    """Create UnitState objects with ai_role and owner set from the army list.
    Heroes with attached_to >= 0 are merged into their host unit's state."""
    td = get_templates_dict()

    # First pass: create states for each resolved unit (1:1 with entries)
    all_states: list[UnitState | None] = []
    for i, entry in enumerate(army.entries):
        us = UnitState(copy.copy(resolved[i]))
        us.ai_role = entry.ai_role
        us.combat_preference = entry.combat_preference
        us.owner = player
        all_states.append(us)

    # Second pass: merge heroes into host units
    hero_indices: set[int] = set()
    for i, entry in enumerate(army.entries):
        if entry.attached_to < 0:
            continue
        tpl = td.get(entry.template_id)
        if not tpl or not tpl.hero:
            continue
        target_idx = entry.attached_to
        if target_idx >= len(all_states) or all_states[target_idx] is None:
            continue
        # Merge hero into host
        merge_hero_into_unit(resolved[i], all_states[target_idx])
        hero_indices.add(i)

    # Return only non-None, non-hero states
    return [s for i, s in enumerate(all_states)
            if i not in hero_indices and s is not None]


def _swiss_pair_round(scores: list[float], played: set[tuple[int, int]],
                      n: int) -> list[tuple[int, int]]:
    """Pair players by similar score, avoiding repeat matchups.

    Sorts by current score descending, then greedily pairs adjacent
    unpaired players, skipping pairs that already played if possible.
    """
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    paired: set[int] = set()
    pairings: list[tuple[int, int]] = []

    for idx in range(len(order)):
        i = order[idx]
        if i in paired:
            continue
        # Try to find closest-ranked unpaired opponent not yet played
        best_j = None
        fallback_j = None
        for jdx in range(idx + 1, len(order)):
            j = order[jdx]
            if j in paired:
                continue
            key = (min(i, j), max(i, j))
            if key not in played:
                best_j = j
                break
            elif fallback_j is None:
                fallback_j = j
        partner = best_j if best_j is not None else fallback_j
        if partner is not None:
            pairings.append((i, partner))
            paired.add(i)
            paired.add(partner)

    return pairings


_WORKER_COUNT = max(1, os.cpu_count() // 2)  # physical cores only (ignore hyperthreads)

# --- ML-enabled evolution worker globals ---
_g_evo_ml_model = None


def _init_evo_worker(use_c_ext: bool = True):
    """Initialize worker process with C extension toggle."""
    import fast_core
    fast_core.USE_C_EXT = use_c_ext and fast_core.is_available()


def _init_evo_ml_worker(model_path: str, model_type: str, use_c_ext: bool = True):
    """Initialize worker process with ML model for evolution games."""
    global _g_evo_ml_model
    import torch
    import fast_core
    fast_core.USE_C_EXT = use_c_ext and fast_core.is_available()
    from ml_model_tactical import TacticalModel
    _g_evo_ml_model = TacticalModel()
    from ml_training import load_model_state_dict
    _g_evo_ml_model.load_state_dict(
        load_model_state_dict(model_path), strict=False)
    _g_evo_ml_model.eval()
    torch.set_num_threads(1)


def _play_matchup(args):
    """Worker function for parallel evaluation. Runs in a subprocess."""
    army_i, army_j, res_i, res_j, mode, games, *rest = args
    use_ml = rest[0] if rest else False
    ml_batch_tactical = rest[1] if len(rest) > 1 else True
    bench = rest[2] if len(rest) > 2 else False
    ml_kw = {}
    if use_ml and _g_evo_ml_model is not None:
        ml_kw = {'ml_model_a': _g_evo_ml_model, 'ml_model_b': _g_evo_ml_model,
                 'ml_batch_tactical': ml_batch_tactical}

    if bench and use_ml and not ml_batch_tactical:
        from ml_integration_tactical import reset_timing, get_timing
        reset_timing()

    a_wins = 0.0
    b_wins = 0.0
    _t_game_logic = 0.0
    for _ in range(games):
        sa = _make_unit_states(army_i, res_i, "A")
        sb = _make_unit_states(army_j, res_j, "B")
        if bench:
            import time
            _gt0 = time.perf_counter()
        result = simulate_game(res_i, res_j, mode=mode, states_a=sa, states_b=sb,
                               **ml_kw)
        if bench:
            _gt1 = time.perf_counter()
            _t_game_logic += _gt1 - _gt0
        if result == 'A':
            a_wins += 1
        elif result == 'B':
            b_wins += 1
        else:
            a_wins += 0.5
            b_wins += 0.5

    if bench and use_ml and not ml_batch_tactical:
        timing = get_timing()
        timing['total_game_s'] = _t_game_logic
        return a_wins, b_wins, timing
    return a_wins, b_wins


def _play_matchup_batched(args):
    """Worker: run a chunk of matchups using batched cross-game inference.

    Uses the two-phase coroutine protocol so each game's destination pointer
    runs against real Dijkstra candidates (not the centroid fallback).
    """
    matchup_list, mode, *rest = args
    games_per_override = rest[0] if rest else None

    from ml_integration_tactical import (
        Phase1Request, Phase2Request,
        batched_phase1_inference, batched_phase2_inference,
    )

    games_per = games_per_override if games_per_override is not None else GAMES_PER_MATCHUP

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
    pending: list[tuple[int, object]] = []
    results = [None] * len(generators)
    for i, gen in enumerate(generators):
        try:
            req = next(gen)
            pending.append((i, req))
        except StopIteration as e:
            results[i] = e.value

    # Per-generator trunk cache, held between Phase 1 and Phase 2.
    trunk_cache: dict[int, dict] = {}

    import torch
    with torch.no_grad():
        while pending:
            # Split by phase. A generator can only be at one phase per tick.
            phase1_pairs = [(i, r) for i, r in pending if isinstance(r, Phase1Request)]
            phase2_pairs = [(i, r) for i, r in pending if isinstance(r, Phase2Request)]
            next_pending: list[tuple[int, object]] = []

            if phase1_pairs:
                p1_reqs = [r for _, r in phase1_pairs]
                p1_results, p1_caches = batched_phase1_inference(
                    _g_evo_ml_model, p1_reqs)
                for (gen_idx, _r), result, cache in zip(phase1_pairs, p1_results, p1_caches):
                    trunk_cache[gen_idx] = cache
                    try:
                        next_req = generators[gen_idx].send(result)
                        next_pending.append((gen_idx, next_req))
                    except StopIteration as e:
                        results[gen_idx] = e.value
                        # Generator finished without yielding phase 2.
                        trunk_cache.pop(gen_idx, None)

            if phase2_pairs:
                p2_reqs = [r for _, r in phase2_pairs]
                caches = [trunk_cache.pop(i) for i, _ in phase2_pairs]
                p2_results = batched_phase2_inference(
                    _g_evo_ml_model, p2_reqs, caches)
                for (gen_idx, _r), result in zip(phase2_pairs, p2_results):
                    try:
                        next_req = generators[gen_idx].send(result)
                        next_pending.append((gen_idx, next_req))
                    except StopIteration as e:
                        results[gen_idx] = e.value

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


def evaluate_population(population: list[ArmyList], mode: str = "objectives",
                        pool: ProcessPoolExecutor | None = None,
                        use_ml: bool = False,
                        ml_batch_tactical: bool = True,
                        bench: bool = False,
                        ml_coroutine_batch: bool = False):
    """Evaluate fitness via Swiss-system tournament.

    Each round pairs players with similar current scores, then all
    games for that round run in parallel.  After SWISS_ROUNDS rounds
    the ranking is stable enough for selection.
    If bench=True, collect and print timing breakdown after first Swiss round.
    """
    n = len(population)
    for ind in population:
        ind.wins = 0.0
        ind.games = 0

    # Pre-resolve all armies once
    resolved = [resolve_army(a) for a in population]

    scores = [0.0] * n
    played: set[tuple[int, int]] = set()

    for rnd in range(SWISS_ROUNDS):
        if rnd == 0:
            # Round 1: random pairing
            order = list(range(n))
            random.shuffle(order)
            round_pairs = [(order[i], order[i + 1])
                           for i in range(0, n - 1, 2)]
        else:
            round_pairs = _swiss_pair_round(scores, played, n)

        if not round_pairs:
            break

        # Record pairings
        for i, j in round_pairs:
            played.add((min(i, j), max(i, j)))

        if ml_coroutine_batch:
            # Chunk matchups across workers for batched cross-game inference
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

            # Unpack results back to per-pair
            for p_idx, (i, j) in enumerate(round_pairs):
                c_idx, pos = pair_chunk_map[p_idx]
                a_wins, b_wins = chunk_results[c_idx][pos]
                population[i].wins += a_wins
                population[i].games += GAMES_PER_MATCHUP
                population[j].wins += b_wins
                population[j].games += GAMES_PER_MATCHUP
                scores[i] = population[i].wins / max(population[i].games, 1)
                scores[j] = population[j].wins / max(population[j].games, 1)
        else:
            # Existing individual matchup path
            # bench flag only on first Swiss round to avoid ongoing overhead
            _bench_this_round = bench and rnd == 0
            work = [(population[i], population[j], resolved[i], resolved[j],
                     mode, GAMES_PER_MATCHUP, use_ml, ml_batch_tactical,
                     _bench_this_round)
                    for i, j in round_pairs]

            if pool is not None:
                results = list(pool.map(_play_matchup, work, chunksize=10))
            else:
                with ProcessPoolExecutor(max_workers=_WORKER_COUNT) as _pool:
                    results = list(_pool.map(_play_matchup, work, chunksize=10))

            # Aggregate timing from first Swiss round
            if _bench_this_round and use_ml and not ml_batch_tactical:
                agg = {'encode_s': 0.0, 'forward_s': 0.0, 'calls': 0,
                       'total_game_s': 0.0}
                for r in results:
                    if len(r) == 3:
                        for k in agg:
                            agg[k] += r[2][k]
                ml_s = agg['encode_s'] + agg['forward_s']
                game_only_s = agg['total_game_s'] - ml_s
                print(f"\n=== TIMING BENCHMARK (Swiss round 1, "
                      f"{len(round_pairs)} matchups × {GAMES_PER_MATCHUP} games) ===")
                print(f"  Total game time (wall, summed across workers): "
                      f"{agg['total_game_s']:.3f}s")
                print(f"  ML encode time:  {agg['encode_s']:.3f}s "
                      f"({100*agg['encode_s']/max(agg['total_game_s'],1e-9):.1f}%)")
                print(f"  ML forward time: {agg['forward_s']:.3f}s "
                      f"({100*agg['forward_s']/max(agg['total_game_s'],1e-9):.1f}%)")
                print(f"  Game logic time: {game_only_s:.3f}s "
                      f"({100*game_only_s/max(agg['total_game_s'],1e-9):.1f}%)")
                print(f"  Forward calls:   {agg['calls']}")
                if agg['calls'] > 0:
                    print(f"  Avg per forward:  {1000*agg['forward_s']/agg['calls']:.3f}ms")
                    print(f"  Avg per encode:   {1000*agg['encode_s']/agg['calls']:.3f}ms")
                print()

            for (i, j), r in zip(round_pairs, results):
                a_wins, b_wins = r[0], r[1]
                population[i].wins += a_wins
                population[i].games += GAMES_PER_MATCHUP
                population[j].wins += b_wins
                population[j].games += GAMES_PER_MATCHUP
                scores[i] = population[i].wins / max(population[i].games, 1)
                scores[j] = population[j].wins / max(population[j].games, 1)

    for ind in population:
        ind.fitness = ind.wins / max(ind.games, 1)


def next_generation(population: list[ArmyList], mode: str = "objectives",
                    enforce_forceorg: bool = False) -> list[ArmyList]:
    """Advance the multi-faction population by one generation.

    Population is partitioned into:
      - meta-chasers (breeder_type='meta'), 75 lists, may switch faction by
        inheriting from a cross-faction tournament winner.
      - hardcore fans (breeder_type='fan_hef' / 'fan_bb' / 'fan_ed'), 25 each,
        whose tournament samples only same-faction parents.

    Per-group elites are preserved (ELITE_META meta + ELITE_PER_FAN per fan
    faction). Each group fills its remaining slots via tournament selection
    inside its allowed parent pool.
    """
    adjusted = _compute_adjusted_fitness(population)

    # Indices grouped by breeder_type
    meta_idxs = [i for i, a in enumerate(population) if a.breeder_type == "meta"]
    fan_idxs: dict[str, list[int]] = {f"fan_{f}": [] for f in FACTIONS}
    for i, a in enumerate(population):
        if a.breeder_type in fan_idxs:
            fan_idxs[a.breeder_type].append(i)

    # ── Per-group elite selection by adjusted fitness ──
    survivors: list[ArmyList] = []
    if meta_idxs:
        meta_sorted = sorted(meta_idxs, key=lambda i: adjusted[i], reverse=True)
        for i in meta_sorted[:ELITE_META]:
            survivors.append(copy.deepcopy(population[i]))
    for fan_key, idxs in fan_idxs.items():
        if not idxs:
            continue
        fan_sorted = sorted(idxs, key=lambda i: adjusted[i], reverse=True)
        for i in fan_sorted[:ELITE_PER_FAN]:
            survivors.append(copy.deepcopy(population[i]))

    # ── Tournament selection for each group ──
    def _pick_parent(pool_idxs: list[int]) -> int:
        k = min(TOURNAMENT_SIZE, len(pool_idxs))
        sampled = random.sample(pool_idxs, k)
        return max(sampled, key=lambda i: adjusted[i])

    def _make_child(parent: ArmyList, breeder_type: str) -> ArmyList:
        child = copy.deepcopy(parent)
        child.breeder_type = breeder_type
        for _ in range(random.randint(1, 4)):
            mutate(child, mode=mode, enforce_forceorg=enforce_forceorg)
        child.fitness = 0.0
        child.wins = 0.0
        child.games = 0
        return child

    offspring: list[ArmyList] = []

    # Meta-chasers: parent pool is the WHOLE population (any breeder, any
    # faction). Faction is inherited from whichever parent wins.
    if meta_idxs:
        whole_pop_idxs = list(range(len(population)))
        for _ in range(META_OFFSPRING):
            p_idx = _pick_parent(whole_pop_idxs)
            offspring.append(_make_child(population[p_idx], "meta"))

    # Hardcore fans: parent pool is restricted to lists of their faction
    # (any breeder type — meta-chasers currently in that faction count).
    for f in FACTIONS:
        fan_key = f"fan_{f}"
        if not fan_idxs[fan_key]:
            continue
        same_faction_pool = [i for i, a in enumerate(population)
                              if a.faction == f]
        if not same_faction_pool:
            # Degenerate case: no same-faction candidate. Fall back to
            # the fan's own elite slot to avoid stalling.
            same_faction_pool = fan_idxs[fan_key]
        for _ in range(FAN_OFFSPRING_PER_FACTION):
            p_idx = _pick_parent(same_faction_pool)
            offspring.append(_make_child(population[p_idx], fan_key))

    return survivors + offspring


def faction_share(population: list[ArmyList],
                  breeder_type: str = "meta") -> dict[str, int]:
    """Count how many lists in *population* of the given breeder_type belong
    to each faction. Used for the per-generation meta-chaser print line.
    Returns a dict {faction: count} with all factions present."""
    out = {f: 0 for f in FACTIONS}
    for a in population:
        if a.breeder_type == breeder_type and a.faction in out:
            out[a.faction] += 1
    return out


if __name__ == "__main__":
    print("wrong python file you muppet")
