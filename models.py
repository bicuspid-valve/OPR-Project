"""Domain model: dataclasses, dice engine, unit resolution and validation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


# ===================================================================
# DATACLASSES
# ===================================================================

@dataclass(frozen=True)
class Weapon:
    name: str
    range_inches: int
    attacks: int
    ap: int = 0
    blast: int = 0
    deadly: int = 0
    crack: bool = False
    rending: bool = False
    reliable: bool = False
    takedown: bool = False
    unstoppable: bool = False
    melee: bool = False
    bane: bool = False
    thrust: bool = False


@dataclass
class UpgradeOption:
    id: str
    cost: int
    description: str = ""
    removes_weapon: str = ""
    removes_count: int = 1
    adds_weapons: list[Weapon] = field(default_factory=list)
    adds_tough: int = 0
    adds_regeneration: bool = False
    adds_piercing_spotter: bool = False
    applies_to_all: bool = False
    # Second weapon removal (for multi-weapon-removal upgrades like Gliders)
    removes_weapon_2: str = ""
    removes_count_2: int = 0
    adds_weapons_2: list[Weapon] = field(default_factory=list)
    requires: str = ""
    # Special-rule-granting fields
    adds_stealth: bool = False
    adds_stealth_aura: bool = False
    adds_scout: bool = False
    adds_scout_aura: bool = False
    adds_fortified: bool = False
    adds_fast: bool = False
    adds_flying: bool = False
    adds_teleport: bool = False
    adds_relentless: bool = False
    adds_fear: int = 0
    removes_shielded: bool = False


@dataclass
class UpgradeSlot:
    id: str
    description: str = ""
    options: list[UpgradeOption] = field(default_factory=list)


@dataclass
class UnitTemplate:
    id: str
    name: str
    base_cost: int
    size: int
    quality: int
    defense: int
    tough: int = 0
    fearless: bool = False
    regeneration: bool = False
    base_weapons: list[Weapon] = field(default_factory=list)
    upgrade_slots: list[UpgradeSlot] = field(default_factory=list)
    # Special rules for grid sim
    scout: bool = False
    stealth: bool = False
    relentless: bool = False
    fast: bool = False
    artillery: bool = False
    # Melee rules
    shielded: bool = False
    furious: bool = False
    impact: int = 0
    fortified: bool = False
    hero: bool = False
    flying: bool = False
    teleport: bool = False
    fear: int = 0
    is_combined: bool = False
    source_template_id: str = ""


@dataclass
class ResolvedUnit:
    template_id: str
    name: str
    models: int
    quality: int
    defense: int
    tough: int = 0
    fearless: bool = False
    regeneration: bool = False
    piercing_spotter: bool = False
    weapons: list[Weapon] = field(default_factory=list)
    weapons_per_model: list[list[Weapon]] = field(default_factory=list)
    points: int = 0
    # Special rules for grid sim
    scout: bool = False
    stealth: bool = False
    relentless: bool = False
    fast: bool = False
    highborn: bool = True  # all HEF units
    artillery: bool = False
    # Melee rules
    shielded: bool = False
    furious: bool = False
    impact: int = 0
    fortified: bool = False
    hero: bool = False
    flying: bool = False
    teleport: bool = False
    fear: int = 0
    stealth_aura: bool = False
    scout_aura: bool = False

    @property
    def label(self) -> str:
        return f"{self.name} [{self.points}pts]"

    @property
    def advance_distance(self) -> int:
        if self.artillery:
            return 0
        d = 6 + 2  # base + Highborn
        if self.fast:
            d += 2
        if self.teleport:
            d += 6
        return d

    @property
    def rush_distance(self) -> int:
        if self.artillery:
            return 0
        d = 12 + 2  # base + Highborn
        if self.fast:
            d += 4
        if self.teleport:
            d += 6
        return d

    @property
    def max_weapon_range(self) -> int:
        if not self.weapons:
            return 0
        return max(w.range_inches for w in self.weapons)


# Army list entry = template + chosen upgrades + AI role
@dataclass
class ArmyListEntry:
    template_id: str
    chosen_upgrades: dict[str, str] = field(default_factory=dict)
    ai_role: str = "killer"  # "killer" | "objective_clearer" | "objective_holder"
    combat_preference: str = "ranged"  # "melee" | "ranged"
    computed_cost: int = 0
    attached_to: int = -1  # index into ArmyList.entries (-1 = unattached hero)


# The genome
@dataclass
class ArmyList:
    entries: list[ArmyListEntry] = field(default_factory=list)
    fitness: float = 0.0
    wins: float = 0.0
    games: int = 0

    @property
    def total_cost(self) -> int:
        return sum(e.computed_cost for e in self.entries)

    @property
    def total_models(self) -> int:
        from templates import get_templates_dict
        templates = get_templates_dict()
        count = 0
        for e in self.entries:
            count += templates[e.template_id].size
        return count

    def validate_heroes(self) -> bool:
        """Check hero attachments are valid."""
        from templates import get_templates_dict
        templates = get_templates_dict()
        hosted_by: set[int] = set()
        for i, e in enumerate(self.entries):
            if e.attached_to < 0:
                continue
            tpl = templates.get(e.template_id)
            if not tpl or not tpl.hero:
                continue
            if e.attached_to >= len(self.entries):
                return False
            target = self.entries[e.attached_to]
            target_tpl = templates.get(target.template_id)
            if not target_tpl or target_tpl.hero:
                return False
            if e.attached_to in hosted_by:
                return False  # duplicate attachment
            hosted_by.add(e.attached_to)
        return True


# ===================================================================
# DICE ENGINE (used by UnitState)
# ===================================================================

_DICE_POOL_SIZE = 8192
_dice_pool = np.empty(0, dtype=np.int8)
_dice_idx = _DICE_POOL_SIZE


def _roll(n: int = 1) -> np.ndarray:
    """Draw n d6 results from the pre-rolled pool."""
    global _dice_pool, _dice_idx
    if _dice_idx + n > len(_dice_pool):
        _dice_pool = np.random.randint(1, 7, size=max(_DICE_POOL_SIZE, n * 2), dtype=np.int8)
        _dice_idx = 0
    result = _dice_pool[_dice_idx:_dice_idx + n]
    _dice_idx += n
    return result


def _roll1() -> int:
    """Draw a single d6."""
    global _dice_pool, _dice_idx
    if _dice_idx >= len(_dice_pool):
        _dice_pool = np.random.randint(1, 7, size=_DICE_POOL_SIZE, dtype=np.int8)
        _dice_idx = 0
    val = int(_dice_pool[_dice_idx])
    _dice_idx += 1
    return val


# ===================================================================
# UNIT STATE (live game state)
# ===================================================================

@dataclass
class UnitState:
    unit: ResolvedUnit
    models_alive: int = 0
    wounds_per_model: list[int] = field(default_factory=list)
    shaken: bool = False
    morale_checked: bool = False
    activated: bool = False
    fatigued: bool = False
    # Grid sim fields
    ai_role: str = "killer"
    combat_preference: str = "ranged"
    assigned_objective: int = -1  # index into objectives list
    positions: list[tuple[int, int]] = field(default_factory=list)  # (col, row) per model
    weapons_per_model: list[list[Weapon]] = field(default_factory=list)
    _removed_positions: list[tuple[int, int]] = field(default_factory=list)
    owner: str = ""  # "A" or "B"
    movement_stance: str = "normal"  # "kite" | "normal" | "aggressive"
    # Hero fields
    hero_model_index: int = -1  # index of hero model in positions (-1 = no hero)
    hero_unit: ResolvedUnit | None = None  # hero's original resolved stats

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.models_alive = self.unit.models
        self.shaken = False
        self.morale_checked = False
        self.activated = False
        self.fatigued = False
        if self.unit.tough:
            self.wounds_per_model = [0] * self.unit.models
        else:
            self.wounds_per_model = []
        # Deep copy so each game gets fresh weapon lists
        self.weapons_per_model = [list(mw) for mw in self.unit.weapons_per_model]
        self._removed_positions = []

    @property
    def destroyed(self) -> bool:
        return self.models_alive <= 0

    def alive_positions(self) -> list[tuple[int, int]]:
        """Return positions of alive models only."""
        return self.positions

    def rout(self):
        """Destroy all remaining models (melee rout). Buffers positions for board cleanup."""
        self._removed_positions.extend(self.positions)
        self.positions.clear()
        self.wounds_per_model.clear()
        self.weapons_per_model.clear()
        self.models_alive = 0
        self.hero_model_index = -1

    def centre(self) -> tuple[float, float]:
        """Average position of alive models."""
        pos = self.positions
        n = len(pos)
        if not n:
            return (0.0, 0.0)
        sc = sr = 0
        for c, r in pos:
            sc += c
            sr += r
        return (sc / n, sr / n)

    def starting_strength(self) -> int:
        if self.unit.tough and self.unit.models == 1:
            return self.unit.tough
        return self.unit.models

    def current_strength(self) -> int:
        if self.unit.tough and self.unit.models == 1:
            return self.unit.tough - sum(self.wounds_per_model)
        return self.models_alive

    def at_half_or_below(self) -> bool:
        return self.current_strength() <= self.starting_strength() // 2

    def _non_hero_alive(self) -> bool:
        """Check if any non-hero models are still alive."""
        if self.hero_model_index < 0:
            return True  # no hero, all models are non-hero
        for i in range(self.models_alive):
            if i != self.hero_model_index:
                return True
        return False

    def _get_tough_target(self, allow_hero: bool = False) -> int:
        """Find the best model to assign a wound to.
        Skips hero model while non-hero models are alive, unless allow_hero=True."""
        best_idx = -1
        best_wounds = -1
        skip_hero = (self.hero_model_index >= 0
                     and not allow_hero
                     and self._non_hero_alive())
        # Determine tough value per model (hero may have different tough)
        for i, w in enumerate(self.wounds_per_model[:self.models_alive]):
            if skip_hero and i == self.hero_model_index:
                continue
            tough_val = self._model_tough(i)
            if w < tough_val and w > best_wounds:
                best_wounds = w
                best_idx = i
        return best_idx

    def _model_tough(self, idx: int) -> int:
        """Return the Tough value for a specific model index."""
        if idx == self.hero_model_index and self.hero_unit is not None:
            return self.hero_unit.tough if self.hero_unit.tough else self.unit.tough
        return self.unit.tough

    def remove_model(self, idx: int):
        """Remove a model by index — pops from all parallel lists."""
        if idx < 0 or idx >= self.models_alive:
            return
        # Buffer position for board cleanup by _sync_dead_models
        self._removed_positions.append(self.positions[idx])
        # Update hero index
        if self.hero_model_index == idx:
            self.hero_model_index = -1
        elif self.hero_model_index > idx:
            self.hero_model_index -= 1
        # Pop from all parallel lists
        self.positions.pop(idx)
        if self.wounds_per_model:
            self.wounds_per_model.pop(idx)
        if self.weapons_per_model:
            self.weapons_per_model.pop(idx)
        self.models_alive -= 1

    def _pick_non_hero_casualty(self) -> int:
        """Pick a non-hero model to remove. Returns index, or -1 if only hero remains."""
        for i in range(self.models_alive - 1, -1, -1):
            if i != self.hero_model_index:
                return i
        # Only hero left
        return self.hero_model_index if self.models_alive > 0 else -1

    def apply_wounds(self, count: int, ignore_regen: bool = False,
                     allow_hero: bool = False) -> int:
        dealt = 0
        if self.unit.tough:
            for _ in range(count):
                if self.models_alive <= 0:
                    break
                if self.unit.regeneration and not ignore_regen:
                    if _roll1() >= 5:
                        continue
                idx = self._get_tough_target(allow_hero=allow_hero)
                if idx < 0:
                    break
                self.wounds_per_model[idx] += 1
                dealt += 1
                tough_val = self._model_tough(idx)
                if self.wounds_per_model[idx] >= tough_val:
                    self.remove_model(idx)
        else:
            if self.unit.regeneration and not ignore_regen:
                for _ in range(count):
                    if self.models_alive <= 0:
                        break
                    if _roll1() >= 5:
                        continue
                    idx = self.hero_model_index if (allow_hero and self.hero_model_index >= 0) else self._pick_non_hero_casualty()
                    if idx < 0:
                        break
                    self.remove_model(idx)
                    dealt += 1
            else:
                remove = min(count, self.models_alive)
                for _ in range(remove):
                    idx = self.hero_model_index if (allow_hero and self.hero_model_index >= 0) else self._pick_non_hero_casualty()
                    if idx < 0:
                        break
                    self.remove_model(idx)
                dealt = remove
        return dealt

    def apply_deadly_wounds(self, count: int, deadly_mult: int,
                            ignore_regen: bool = False,
                            allow_hero: bool = False) -> int:
        if self.models_alive <= 0:
            return 0
        total = 0
        for _ in range(count):
            if self.models_alive <= 0:
                break
            if self.unit.regeneration and not ignore_regen:
                if _roll1() >= 5:
                    continue
            if self.unit.tough:
                idx = self._get_tough_target(allow_hero=allow_hero)
                if idx < 0:
                    break
                tough_val = self._model_tough(idx)
                remaining_hp = tough_val - self.wounds_per_model[idx]
                actual = min(deadly_mult, remaining_hp)
                self.wounds_per_model[idx] += actual
                total += actual
                if self.wounds_per_model[idx] >= tough_val:
                    self.remove_model(idx)
            else:
                idx = self.hero_model_index if (allow_hero and self.hero_model_index >= 0) else self._pick_non_hero_casualty()
                if idx < 0:
                    break
                self.remove_model(idx)
                total += 1
        return total


# ===================================================================
# HERO MERGING
# ===================================================================

def merge_hero_into_unit(hero_resolved: ResolvedUnit,
                         host_state: UnitState):
    """Merge a hero into a host unit's UnitState.
    Hero model becomes the last model in the unit."""
    # Append hero weapons to unit
    host_state.unit.weapons = list(host_state.unit.weapons) + list(hero_resolved.weapons)
    # Increment model count
    host_state.unit.models += 1
    host_state.models_alive += 1
    # Hero is last model
    host_state.hero_model_index = host_state.unit.models - 1
    # Store hero stats for defense switching
    host_state.hero_unit = hero_resolved
    # Extend wound tracking
    if host_state.unit.tough:
        host_state.wounds_per_model.append(0)
    # Shielded: only applies if both hero and host have it
    if host_state.unit.shielded and not hero_resolved.shielded:
        host_state.unit.shielded = False
    # Aura propagation: hero auras grant rules to the joined unit
    if hero_resolved.stealth_aura:
        host_state.unit.stealth = True
    if hero_resolved.scout_aura:
        host_state.unit.scout = True
    # Append hero's weapons as a new model entry
    host_state.weapons_per_model.append(list(hero_resolved.weapons))


# ===================================================================
# RESOLVE & VALIDATE
# ===================================================================

def _slot_model_index(slot_id: str, size: int) -> int | None:
    """Extract 1-based model index from slot ID suffix like 'sniper_1' → 0.
    Returns None if no suffix, or if the index is out of range for the unit size.
    For combined template _a/_b slots, strips the half suffix first and offsets
    _b indices by size//2."""
    raw = slot_id
    half_offset = 0
    if raw.endswith("_a") or raw.endswith("_b"):
        if raw.endswith("_b"):
            half_offset = size // 2
        raw = raw[:-2]  # strip _a/_b
    m = re.search(r'_(\d+)$', raw)
    if m:
        idx = int(m.group(1)) - 1 + half_offset
        if 0 <= idx < size:
            return idx
    return None


def _apply_weapon_removal(weapons: list[Weapon], weapon_name: str,
                           count: int) -> int:
    """Remove up to `count` weapons matching `weapon_name` from flat list.
    Returns the number actually removed."""
    removed = 0
    new_weapons = []
    for w in weapons:
        if w.name == weapon_name and removed < count:
            removed += 1
        else:
            new_weapons.append(w)
    weapons.clear()
    weapons.extend(new_weapons)
    return removed


def _apply_opt_flat(opt: UpgradeOption, weapons: list[Weapon],
                    is_all_models: bool) -> None:
    """Apply an upgrade option to the flat weapon list.
    For applies_to_all upgrades, adds proportional to actual removals."""
    if opt.removes_weapon:
        before = sum(1 for w in weapons if w.name == opt.removes_weapon)
        _apply_weapon_removal(weapons, opt.removes_weapon,
                              before if is_all_models else opt.removes_count)
        if opt.requires and opt.requires.startswith("sniper_rifle") and opt.removes_weapon == "Mounted Shardguns":
            _apply_weapon_removal(weapons, "Sniper Rifle", 1)
        if is_all_models and opt.adds_weapons:
            actually_removed = before - sum(1 for w in weapons if w.name == opt.removes_weapon)
            replacement = opt.adds_weapons[0]
            weapons.extend([replacement] * actually_removed)
        else:
            weapons.extend(opt.adds_weapons)
    else:
        weapons.extend(opt.adds_weapons)
    # Second weapon removal
    if opt.removes_weapon_2:
        before2 = sum(1 for w in weapons if w.name == opt.removes_weapon_2)
        _apply_weapon_removal(weapons, opt.removes_weapon_2,
                              before2 if is_all_models else opt.removes_count_2)
        if is_all_models and opt.adds_weapons_2:
            actually_removed2 = before2 - sum(1 for w in weapons if w.name == opt.removes_weapon_2)
            replacement2 = opt.adds_weapons_2[0]
            weapons.extend([replacement2] * actually_removed2)
        else:
            weapons.extend(opt.adds_weapons_2)


def _half_model_range(slot_id: str, size: int) -> range:
    """For combined template _a/_b slots, restrict to the appropriate half."""
    if slot_id.endswith("_a"):
        return range(0, size // 2)
    elif slot_id.endswith("_b"):
        return range(size // 2, size)
    return range(size)


def _apply_opt_wpm(opt: UpgradeOption, slot_id: str, wpm: list[list[Weapon]],
                   size: int, is_all_models: bool) -> None:
    """Apply an upgrade option to per-model weapon lists (two-pass aware)."""
    target_model = _slot_model_index(slot_id, size)
    model_range = _half_model_range(slot_id, size)

    def _remove_from_wpm(weapon_name: str, count: int) -> int:
        nonlocal target_model
        removed = 0
        if target_model is not None and target_model in model_range:
            new_mw = []
            for w in wpm[target_model]:
                if w.name == weapon_name and removed < count:
                    removed += 1
                else:
                    new_mw.append(w)
            wpm[target_model] = new_mw
        else:
            target_model = None  # reset if out of range
            for mi in model_range:
                new_mw = []
                for w in wpm[mi]:
                    if w.name == weapon_name and removed < count:
                        removed += 1
                        target_model = mi
                    else:
                        new_mw.append(w)
                wpm[mi] = new_mw
                if removed >= count:
                    break
        return removed

    def _distribute_weapons(weapons: list[Weapon]) -> None:
        """Add weapons to models within model_range."""
        mr = list(model_range)
        if len(weapons) >= len(mr):
            for i, w in enumerate(weapons):
                wpm[mr[i % len(mr)]].append(w)
        else:
            for w in weapons:
                mi = min(mr, key=lambda m: len(wpm[m]))
                wpm[mi].append(w)

    # Primary weapon removal
    if opt.removes_weapon:
        if is_all_models:
            # All-models: remove from each model in range, add proportionally
            total_removed = 0
            for mi in model_range:
                before = sum(1 for w in wpm[mi] if w.name == opt.removes_weapon)
                new_mw = []
                mi_removed = 0
                for w in wpm[mi]:
                    if w.name == opt.removes_weapon and mi_removed < before:
                        mi_removed += 1
                    else:
                        new_mw.append(w)
                wpm[mi] = new_mw
                # Add replacement proportionally per model
                if opt.adds_weapons and mi_removed > 0:
                    replacement = opt.adds_weapons[0]
                    wpm[mi].extend([replacement] * mi_removed)
                total_removed += mi_removed
        else:
            removed = _remove_from_wpm(opt.removes_weapon, opt.removes_count)
            # Handle sniper+shardgun combo
            if (opt.requires and opt.requires.startswith("sniper_rifle")
                    and opt.removes_weapon == "Mounted Shardguns"):
                if target_model is not None:
                    wpm[target_model] = [w for w in wpm[target_model]
                                         if w.name != "Sniper Rifle"]
            # Add weapons to target model
            if opt.adds_weapons:
                if target_model is not None:
                    wpm[target_model].extend(opt.adds_weapons)
                else:
                    _distribute_weapons(opt.adds_weapons)
    elif opt.adds_weapons:
        if is_all_models:
            # All-models add (no removal): distribute evenly within range
            mr = list(model_range)
            if len(opt.adds_weapons) >= len(mr):
                for i, w in enumerate(opt.adds_weapons):
                    wpm[mr[i % len(mr)]].append(w)
            else:
                for w in opt.adds_weapons:
                    mi = min(mr, key=lambda m: len(wpm[m]))
                    wpm[mi].append(w)
        else:
            if target_model is not None and target_model in model_range:
                wpm[target_model].extend(opt.adds_weapons)
            else:
                _distribute_weapons(opt.adds_weapons)

    # Second weapon removal
    if opt.removes_weapon_2:
        if is_all_models:
            for mi in model_range:
                before2 = sum(1 for w in wpm[mi] if w.name == opt.removes_weapon_2)
                new_mw = []
                mi_removed = 0
                for w in wpm[mi]:
                    if w.name == opt.removes_weapon_2 and mi_removed < before2:
                        mi_removed += 1
                    else:
                        new_mw.append(w)
                wpm[mi] = new_mw
                if opt.adds_weapons_2 and mi_removed > 0:
                    replacement2 = opt.adds_weapons_2[0]
                    wpm[mi].extend([replacement2] * mi_removed)
        else:
            saved_target = target_model
            target_model = saved_target  # use same model as primary removal
            _remove_from_wpm(opt.removes_weapon_2, opt.removes_count_2)
            if opt.adds_weapons_2:
                if target_model is not None:
                    wpm[target_model].extend(opt.adds_weapons_2)


def resolve_entry(entry: ArmyListEntry) -> ResolvedUnit:
    """Convert an ArmyListEntry into a fully resolved unit for simulation.
    Uses two-pass ordering: per-model upgrades first, then all-models upgrades."""
    from templates import get_templates_dict
    tpl = get_templates_dict()[entry.template_id]
    weapons = list(tpl.base_weapons)
    tough = tpl.tough
    fearless = tpl.fearless
    regen = tpl.regeneration
    spotter = False
    cost = tpl.base_cost
    upgrade_names: list[str] = []
    # Mutable rule flags (may be modified by upgrades)
    scout = tpl.scout
    stealth = tpl.stealth
    relentless = tpl.relentless
    fast = tpl.fast
    shielded = tpl.shielded
    fortified = tpl.fortified
    flying = tpl.flying
    teleport = tpl.teleport
    fear = tpl.fear
    stealth_aura = False
    scout_aura = False
    impact = tpl.impact

    # Collect chosen options with their slot IDs
    chosen: list[tuple[str, UpgradeOption]] = []
    for slot in tpl.upgrade_slots:
        if slot.id in entry.chosen_upgrades:
            opt_id = entry.chosen_upgrades[slot.id]
            for opt in slot.options:
                if opt.id == opt_id:
                    chosen.append((slot.id, opt))
                    break

    # Separate into per-model (pass 1) and all-models (pass 2)
    per_model = [(sid, opt) for sid, opt in chosen if not opt.applies_to_all]
    all_models = [(sid, opt) for sid, opt in chosen if opt.applies_to_all]

    def _apply_rule_flags(opt: UpgradeOption) -> None:
        nonlocal tough, regen, spotter, scout, stealth, relentless, fast
        nonlocal shielded, fortified, flying, teleport, fear
        nonlocal stealth_aura, scout_aura
        if opt.adds_tough:
            tough = opt.adds_tough
        if opt.adds_regeneration:
            regen = True
        if opt.adds_piercing_spotter:
            spotter = True
        if opt.adds_stealth:
            stealth = True
        if opt.adds_stealth_aura:
            if tpl.hero:
                stealth_aura = True
            else:
                stealth = True
        if opt.adds_scout:
            scout = True
        if opt.adds_scout_aura:
            if tpl.hero:
                scout_aura = True
            else:
                scout = True
        if opt.adds_fortified:
            fortified = True
        if opt.adds_fast:
            fast = True
        if opt.adds_flying:
            flying = True
        if opt.adds_teleport:
            teleport = True
        if opt.adds_relentless:
            relentless = True
        if opt.adds_fear:
            fear = opt.adds_fear

    # --- Flat weapon list (two-pass) ---
    for _, opt in per_model:
        cost += opt.cost
        upgrade_names.append(opt.id)
        _apply_opt_flat(opt, weapons, is_all_models=False)
        _apply_rule_flags(opt)

    for _, opt in all_models:
        cost += opt.cost
        upgrade_names.append(opt.id)
        _apply_opt_flat(opt, weapons, is_all_models=True)
        _apply_rule_flags(opt)

    # removes_shielded: run after all upgrades (any ranged weapon strips shield)
    if shielded:
        for _, opt in per_model + all_models:
            if opt.removes_shielded:
                shielded = False
                break

    # --- Build per-model weapon assignment (two-pass) ---
    wpm: list[list[Weapon]] = [[] for _ in range(tpl.size)]

    # Base weapon distribution: round-robin across models
    for i, w in enumerate(tpl.base_weapons):
        wpm[i % tpl.size].append(w)

    # Pass 1: per-model upgrades
    for slot_id, opt in per_model:
        _apply_opt_wpm(opt, slot_id, wpm, tpl.size, is_all_models=False)

    # Pass 2: all-models upgrades
    for slot_id, opt in all_models:
        _apply_opt_wpm(opt, slot_id, wpm, tpl.size, is_all_models=True)

    name_parts = [tpl.name]
    if upgrade_names:
        name_parts.append("(" + ", ".join(upgrade_names) + ")")
    display_name = " ".join(name_parts)

    entry.computed_cost = cost
    return ResolvedUnit(
        template_id=tpl.id,
        name=display_name,
        models=tpl.size,
        quality=tpl.quality,
        defense=tpl.defense,
        tough=tough,
        fearless=fearless,
        regeneration=regen,
        piercing_spotter=spotter,
        weapons=weapons,
        weapons_per_model=wpm,
        points=cost,
        scout=scout,
        stealth=stealth,
        relentless=relentless,
        fast=fast,
        highborn=True,
        artillery=tpl.artillery,
        shielded=shielded,
        furious=tpl.furious,
        impact=impact,
        fortified=fortified,
        hero=tpl.hero,
        flying=flying,
        teleport=teleport,
        fear=fear,
        stealth_aura=stealth_aura,
        scout_aura=scout_aura,
    )


def compute_entry_cost(entry: ArmyListEntry) -> int:
    """Compute cost without full resolution."""
    from templates import get_templates_dict
    tpl = get_templates_dict()[entry.template_id]
    cost = tpl.base_cost
    for slot in tpl.upgrade_slots:
        if slot.id in entry.chosen_upgrades:
            opt_id = entry.chosen_upgrades[slot.id]
            for opt in slot.options:
                if opt.id == opt_id:
                    cost += opt.cost
                    break
    entry.computed_cost = cost
    return cost


def validate_upgrades(entry: ArmyListEntry) -> bool:
    """Check that chosen upgrades are legal."""
    from templates import get_templates_dict
    tpl = get_templates_dict().get(entry.template_id)
    if not tpl:
        return False
    chosen_ids = set()
    for slot in tpl.upgrade_slots:
        if slot.id in entry.chosen_upgrades:
            opt_id = entry.chosen_upgrades[slot.id]
            found = False
            for opt in slot.options:
                if opt.id == opt_id:
                    found = True
                    chosen_ids.add(opt_id)
                    if opt.requires and opt.requires not in chosen_ids:
                        req_found = False
                        for s2 in tpl.upgrade_slots:
                            if s2.id in entry.chosen_upgrades:
                                if entry.chosen_upgrades[s2.id] == opt.requires:
                                    req_found = True
                                    break
                        if not req_found:
                            return False
                    break
            if not found:
                return False
    return True
