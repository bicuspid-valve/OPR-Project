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
    # Battle Brothers additions
    lacerate: bool = False     # defender re-rolls successful blocks
    shred: bool = False        # unmodified def roll of 1 → +1 wound
    smash: bool = False        # gains Blast(3) vs targets where >50% models have Def 5+
    indirect: bool = False     # ignores line-of-sight / cover
    limited: bool = False      # only fires on the unit's first activation in the game
    # Eternal Dynasty additions
    surge: bool = False        # unmodified roll of 6 to hit → +1 extra hit
    tear: bool = False         # AP(+4) vs targets where most models have Tough(9)+
    puncture: bool = False     # ignores Regen, AP(+4) vs Tough(3)–Tough(9)
    # Terrain spec §4.5: per-weapon flag suppressing the +1 cover defense bonus.
    # OR'd with unit-level ignores_cover at resolution time.
    ignores_cover: bool = False


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
    # Battle Brothers additions
    adds_strider: bool = False
    adds_versatile_attack: bool = False
    adds_versatile_reach: bool = False
    adds_unstoppable_mark: bool = False
    adds_impact: int = 0
    adds_shielded: bool = False
    # Auras (granted by hero upgrades; propagate to host unit on merge)
    adds_bane_melee_aura: bool = False
    adds_bane_shoot_aura: bool = False
    adds_courage_aura: bool = False
    adds_rapid_rush_aura: bool = False
    adds_regeneration_aura: bool = False
    adds_versatile_reach_aura: bool = False
    # Eternal Dynasty additions
    adds_piercing_hunter: bool = False
    adds_melee_evasion: bool = False
    adds_counter_attack: bool = False
    adds_unpredictable_fighter: bool = False
    adds_rapid_advance: bool = False
    adds_rapid_charge: bool = False
    adds_bounding: bool = False
    adds_ed_teleport: bool = False
    adds_vengeance: bool = False
    adds_isr_mark: bool = False
    adds_ignores_cover: bool = False
    adds_fearless: bool = False
    adds_furious: bool = False
    # ED auras
    adds_clan_warrior_boost_aura: bool = False
    adds_counter_attack_aura: bool = False
    adds_fearless_aura: bool = False
    adds_ignores_cover_aura: bool = False
    adds_melee_evasion_aura: bool = False
    adds_piercing_hunter_aura: bool = False
    adds_precision_fighter_aura: bool = False    # +1 to hit in melee for the unit
    adds_rapid_advance_aura: bool = False
    adds_rapid_charge_aura: bool = False
    adds_stealth_aura_ed: bool = False            # alias — same effect as existing stealth aura


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
    # Faction: "hef" (High Elf Fleet, +2" Highborn movement),
    # "bb" (Battle Brothers, Battleborn shaken auto-recovery), or
    # "ed" (Eternal Dynasty, army-wide Clan Warrior nat-6 hit explosion).
    faction: str = "hef"
    # Battle Brothers unit-level rules
    battleborn: bool = False
    strider: bool = False
    versatile_attack: bool = False     # per-activation EV-pick of AP(+1) vs +1-to-hit at >9"
    versatile_reach: bool = False      # permanent +4" range (shoot) and +2" charge
    unstoppable_mark: bool = False     # once per activation: mark enemy → allies get Unstoppable
    # Eternal Dynasty unit-level rules
    clan_warrior: bool = False              # army-wide: nat 6 to hit → +1 extra attack
    clan_warrior_boost: bool = False        # extra attacks trigger on 5-6 instead of just 6
    piercing_hunter: bool = False           # +1 AP when shooting >9"
    melee_evasion: bool = False             # melee attackers get -1 to hit
    counter_attack: bool = False            # strikes first when charged
    unpredictable_fighter: bool = False     # melee: roll d6, 1-3 AP+1, 4-6 +1 hit
    rapid_advance: bool = False             # +4" Advance
    rapid_charge: bool = False              # +4" Charge
    bounding: bool = False                  # at activation, place models within D3+1"
    ed_teleport: bool = False               # per-activation 3"/3"/6" reposition trigger
    vengeance: bool = False                 # leaves Vengeance markers when destroyed
    isr_mark: bool = False                  # once per activation: mark enemy → +6" range
    ignores_cover: bool = False             # ignores stealth/cover when shooting
    slow: bool = False                      # cannot Rush
    precision_fighter: bool = False         # +1 to hit in melee
    # Bane / courage / rapid_rush moved here at template level for parity
    # (legacy: only flowed via aura; templates don't normally set these directly).


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
    artillery: bool = False
    # Faction: drives faction-wide rules. Highborn (HEF) gives +2" advance/rush;
    # Battleborn (BB) gives a 4+ shaken auto-recovery at the start of each round.
    faction: str = "hef"
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
    # Battle Brothers unit-level rules
    battleborn: bool = False
    strider: bool = False
    versatile_attack: bool = False
    versatile_reach: bool = False
    unstoppable_mark: bool = False
    # Eternal Dynasty unit-level rules
    clan_warrior: bool = False
    clan_warrior_boost: bool = False
    piercing_hunter: bool = False
    melee_evasion: bool = False
    counter_attack: bool = False
    unpredictable_fighter: bool = False
    rapid_advance: bool = False
    rapid_charge: bool = False
    bounding: bool = False
    ed_teleport: bool = False
    vengeance: bool = False
    isr_mark: bool = False
    ignores_cover: bool = False
    slow: bool = False
    precision_fighter: bool = False
    # Effective rules after aura propagation
    bane_melee: bool = False
    bane_shoot: bool = False
    courage: bool = False
    rapid_rush: bool = False
    # Aura sources (set on heroes; transferred to host on merge)
    bane_melee_aura: bool = False
    bane_shoot_aura: bool = False
    courage_aura: bool = False
    rapid_rush_aura: bool = False
    regeneration_aura: bool = False
    versatile_reach_aura: bool = False
    # ED auras
    clan_warrior_boost_aura: bool = False
    counter_attack_aura: bool = False
    fearless_aura: bool = False
    ignores_cover_aura: bool = False
    melee_evasion_aura: bool = False
    piercing_hunter_aura: bool = False
    precision_fighter_aura: bool = False
    rapid_advance_aura: bool = False
    rapid_charge_aura: bool = False

    @property
    def label(self) -> str:
        return f"{self.name} [{self.points}pts]"

    @property
    def highborn(self) -> bool:
        return self.faction == "hef"

    @property
    def advance_distance(self) -> int:
        if self.artillery:
            return 0
        d = 6
        if self.faction == "hef":
            d += 2  # Highborn
        if self.fast:
            d += 2
        if self.teleport or self.ed_teleport:
            d += 6
        if self.rapid_advance:
            d += 4
        if self.bounding:
            d += 3  # Bounding D3+1 reposition averages ~3"
        return d

    @property
    def rush_distance(self) -> int:
        if self.artillery:
            return 0
        if self.slow:
            # Slow units cannot Rush — fall back to their Advance distance.
            return self.advance_distance
        d = 12
        if self.faction == "hef":
            d += 2  # Highborn
        if self.fast:
            d += 4
        if self.teleport or self.ed_teleport:
            d += 6
        if self.rapid_rush:
            d += 6
        if self.bounding:
            d += 3
        return d

    @property
    def charge_distance(self) -> int:
        # Charge uses Rush distance; Versatile Reach grants +2" when charging.
        # Rapid Charge stacks +4" on Charge specifically.
        if self.artillery:
            return 0
        d = self.rush_distance
        if self.versatile_reach:
            d += 2
        if self.rapid_charge:
            d += 4
        return d

    @property
    def max_weapon_range(self) -> int:
        if not self.weapons:
            return 0
        base = max(w.range_inches for w in self.weapons)
        # Versatile Reach: +4" to all ranged weapons in this unit.
        if self.versatile_reach and base > 0:
            base += 4
        return base


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
    # Faction is "hef" (default — preserves backward compatibility for
    # existing HEF training runs), "bb", or "ed". An ArmyList is homogeneous.
    faction: str = "hef"
    # Breeder type for multi-faction evolution:
    #   "meta"      — meta-chaser; tournament samples across the whole
    #                 population and may switch faction by inheritance.
    #   "fan_hef" / "fan_bb" / "fan_ed" — hardcore fan; tournament samples
    #                 only same-faction parents and never switches faction.
    breeder_type: str = "meta"

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
    # Battle Brothers turn-state
    has_activated_once: bool = False     # true after the unit's first activation begins
    limited_spent: bool = False          # Limited weapons removed after first activation
    unstoppable_mark_used: bool = False  # once-per-activation cap for Unstoppable Mark
    marked_by_unstoppable: bool = False  # set on a target when an enemy uses the mark
    # Eternal Dynasty turn-state
    isr_mark_used: bool = False          # once-per-activation cap for Increased Shooting Range Mark
    marked_by_isr: bool = False          # set on target — friendly side gets +6" range once
    marked_by_isr_owner: str = ""        # which owner ("A"/"B") set the mark
    vengeance_markers: int = 0           # markers placed on this unit by destroyed Vengeance units

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
        # BB state
        self.has_activated_once = False
        self.limited_spent = False
        self.unstoppable_mark_used = False
        self.marked_by_unstoppable = False
        # ED state
        self.isr_mark_used = False
        self.marked_by_isr = False
        self.marked_by_isr_owner = ""
        self.vengeance_markers = 0

    def has_limited_weapon(self) -> bool:
        if self.limited_spent:
            return False
        for mw in self.weapons_per_model:
            for w in mw:
                if w.limited:
                    return True
        return False

    def consume_limited(self) -> bool:
        """Strip Limited weapons from per-model loadout after the first activation.
        Returns True iff anything was removed (caller may want to re-encode features)."""
        if self.limited_spent:
            return False
        removed = False
        for i, mw in enumerate(self.weapons_per_model):
            if any(w.limited for w in mw):
                self.weapons_per_model[i] = [w for w in mw if not w.limited]
                removed = True
        self.limited_spent = True
        return removed

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
    # Battle Brothers auras
    if hero_resolved.bane_melee_aura:
        host_state.unit.bane_melee = True
    if hero_resolved.bane_shoot_aura:
        host_state.unit.bane_shoot = True
    if hero_resolved.courage_aura:
        host_state.unit.courage = True
    if hero_resolved.rapid_rush_aura:
        host_state.unit.rapid_rush = True
    if hero_resolved.regeneration_aura:
        host_state.unit.regeneration = True
    if hero_resolved.versatile_reach_aura:
        host_state.unit.versatile_reach = True
    # Eternal Dynasty auras
    if hero_resolved.clan_warrior_boost_aura:
        host_state.unit.clan_warrior_boost = True
    if hero_resolved.counter_attack_aura:
        host_state.unit.counter_attack = True
    if hero_resolved.fearless_aura:
        host_state.unit.fearless = True
    if hero_resolved.ignores_cover_aura:
        host_state.unit.ignores_cover = True
    if hero_resolved.melee_evasion_aura:
        host_state.unit.melee_evasion = True
    if hero_resolved.piercing_hunter_aura:
        host_state.unit.piercing_hunter = True
    if hero_resolved.precision_fighter_aura:
        host_state.unit.precision_fighter = True
    if hero_resolved.rapid_advance_aura:
        host_state.unit.rapid_advance = True
    if hero_resolved.rapid_charge_aura:
        host_state.unit.rapid_charge = True
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
    For applies_to_all upgrades, adds proportional to actual removals
    when adds_weapons is a single per-model replacement; if the add list
    encodes multiple weapons per removal (len divisible by removed count),
    the full list is added verbatim."""
    if opt.removes_weapon:
        before = sum(1 for w in weapons if w.name == opt.removes_weapon)
        _apply_weapon_removal(weapons, opt.removes_weapon,
                              before if is_all_models else opt.removes_count)
        if opt.requires and opt.requires.startswith("sniper_rifle") and opt.removes_weapon == "Mounted Shardguns":
            _apply_weapon_removal(weapons, "Sniper Rifle", 1)
        if is_all_models and opt.adds_weapons:
            actually_removed = before - sum(1 for w in weapons if w.name == opt.removes_weapon)
            n_add = len(opt.adds_weapons)
            if actually_removed > 0 and n_add % actually_removed == 0:
                weapons.extend(opt.adds_weapons)
            else:
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
            n_add2 = len(opt.adds_weapons_2)
            if actually_removed2 > 0 and n_add2 % actually_removed2 == 0:
                weapons.extend(opt.adds_weapons_2)
            else:
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
            mr = list(model_range)
            n_models = len(mr)
            # If adds_weapons specifies multiple weapons per model
            # (len divisible by n_models), distribute round-robin so each
            # model gets len/n_models weapons. Else fall back to the
            # legacy single-replacement-scaled-by-removals behavior.
            distribute_full = (opt.adds_weapons and n_models > 0
                               and len(opt.adds_weapons) % n_models == 0)
            for mi_idx, mi in enumerate(mr):
                before = sum(1 for w in wpm[mi] if w.name == opt.removes_weapon)
                new_mw = []
                mi_removed = 0
                for w in wpm[mi]:
                    if w.name == opt.removes_weapon and mi_removed < before:
                        mi_removed += 1
                    else:
                        new_mw.append(w)
                wpm[mi] = new_mw
                if distribute_full:
                    wpm[mi].extend(opt.adds_weapons[mi_idx::n_models])
                elif opt.adds_weapons and mi_removed > 0:
                    replacement = opt.adds_weapons[0]
                    wpm[mi].extend([replacement] * mi_removed)
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
            mr2 = list(model_range)
            n_models2 = len(mr2)
            distribute_full2 = (opt.adds_weapons_2 and n_models2 > 0
                                and len(opt.adds_weapons_2) % n_models2 == 0)
            for mi_idx, mi in enumerate(mr2):
                before2 = sum(1 for w in wpm[mi] if w.name == opt.removes_weapon_2)
                new_mw = []
                mi_removed = 0
                for w in wpm[mi]:
                    if w.name == opt.removes_weapon_2 and mi_removed < before2:
                        mi_removed += 1
                    else:
                        new_mw.append(w)
                wpm[mi] = new_mw
                if distribute_full2:
                    wpm[mi].extend(opt.adds_weapons_2[mi_idx::n_models2])
                elif opt.adds_weapons_2 and mi_removed > 0:
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
    # Battle Brothers
    strider = tpl.strider
    versatile_attack = tpl.versatile_attack
    versatile_reach = tpl.versatile_reach
    unstoppable_mark = tpl.unstoppable_mark
    bane_melee_aura = False
    bane_shoot_aura = False
    courage_aura = False
    rapid_rush_aura = False
    regeneration_aura = False
    versatile_reach_aura = False
    # Non-hero "banner aura" effective fields (applied directly to the unit)
    bane_melee_local = False
    bane_shoot_local = False
    courage_local = False
    rapid_rush_local = False
    # Eternal Dynasty
    piercing_hunter = tpl.piercing_hunter
    melee_evasion = tpl.melee_evasion
    counter_attack = tpl.counter_attack
    unpredictable_fighter = tpl.unpredictable_fighter
    rapid_advance = tpl.rapid_advance
    rapid_charge = tpl.rapid_charge
    bounding = tpl.bounding
    ed_teleport = tpl.ed_teleport
    vengeance = tpl.vengeance
    isr_mark = tpl.isr_mark
    ignores_cover = tpl.ignores_cover
    precision_fighter = tpl.precision_fighter
    clan_warrior_boost = tpl.clan_warrior_boost
    clan_warrior_boost_aura = False
    counter_attack_aura = False
    fearless_aura = False
    ignores_cover_aura = False
    melee_evasion_aura = False
    piercing_hunter_aura = False
    precision_fighter_aura = False
    rapid_advance_aura = False
    rapid_charge_aura = False

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
        nonlocal shielded, fortified, flying, teleport, fear, fearless, impact
        nonlocal stealth_aura, scout_aura
        nonlocal strider, versatile_attack, versatile_reach, unstoppable_mark
        nonlocal bane_melee_aura, bane_shoot_aura, courage_aura
        nonlocal rapid_rush_aura, regeneration_aura, versatile_reach_aura
        # ED nonlocals
        nonlocal piercing_hunter, melee_evasion, counter_attack
        nonlocal unpredictable_fighter, rapid_advance, rapid_charge
        nonlocal bounding, ed_teleport, vengeance, isr_mark, ignores_cover
        nonlocal precision_fighter, clan_warrior_boost
        nonlocal clan_warrior_boost_aura, counter_attack_aura, fearless_aura
        nonlocal ignores_cover_aura, melee_evasion_aura, piercing_hunter_aura
        nonlocal precision_fighter_aura, rapid_advance_aura, rapid_charge_aura
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
        if opt.adds_impact:
            impact = opt.adds_impact
        if opt.adds_shielded:
            shielded = True
        # Battle Brothers
        if opt.adds_strider:
            strider = True
        if opt.adds_versatile_attack:
            versatile_attack = True
        if opt.adds_versatile_reach:
            versatile_reach = True
        if opt.adds_unstoppable_mark:
            unstoppable_mark = True
        # Auras granted by heroes are stored as aura fields and propagate on
        # attach. Auras granted by banner-style upgrades on non-hero units
        # apply directly to the unit (one model carries the banner; we
        # collapse this to a unit-wide flag for simplicity).
        nonlocal bane_melee_local, bane_shoot_local, courage_local, rapid_rush_local
        if opt.adds_bane_melee_aura:
            if tpl.hero:
                bane_melee_aura = True
            else:
                bane_melee_local = True
        if opt.adds_bane_shoot_aura:
            if tpl.hero:
                bane_shoot_aura = True
            else:
                bane_shoot_local = True
        if opt.adds_courage_aura:
            if tpl.hero:
                courage_aura = True
            else:
                courage_local = True
        if opt.adds_rapid_rush_aura:
            if tpl.hero:
                rapid_rush_aura = True
            else:
                rapid_rush_local = True
        if opt.adds_regeneration_aura:
            if tpl.hero:
                regeneration_aura = True
            else:
                regen = True
        if opt.adds_versatile_reach_aura:
            if tpl.hero:
                versatile_reach_aura = True
            else:
                versatile_reach = True
        # ED unit-level rule grants
        if opt.adds_piercing_hunter:
            piercing_hunter = True
        if opt.adds_melee_evasion:
            melee_evasion = True
        if opt.adds_counter_attack:
            counter_attack = True
        if opt.adds_unpredictable_fighter:
            unpredictable_fighter = True
        if opt.adds_rapid_advance:
            rapid_advance = True
        if opt.adds_rapid_charge:
            rapid_charge = True
        if opt.adds_bounding:
            bounding = True
        if opt.adds_ed_teleport:
            ed_teleport = True
        if opt.adds_vengeance:
            vengeance = True
        if opt.adds_isr_mark:
            isr_mark = True
        if opt.adds_ignores_cover:
            ignores_cover = True
        if opt.adds_fearless:
            fearless = True
        # ED auras — propagate to a host on attach when the model is a hero;
        # for a non-hero model the upgrade is treated as directly applying
        # the aura's effect to the unit.
        if opt.adds_clan_warrior_boost_aura:
            if tpl.hero:
                clan_warrior_boost_aura = True
            else:
                clan_warrior_boost = True
        if opt.adds_counter_attack_aura:
            if tpl.hero:
                counter_attack_aura = True
            else:
                counter_attack = True
        if opt.adds_fearless_aura:
            if tpl.hero:
                fearless_aura = True
            else:
                fearless = True
        if opt.adds_ignores_cover_aura:
            if tpl.hero:
                ignores_cover_aura = True
            else:
                ignores_cover = True
        if opt.adds_melee_evasion_aura:
            if tpl.hero:
                melee_evasion_aura = True
            else:
                melee_evasion = True
        if opt.adds_piercing_hunter_aura:
            if tpl.hero:
                piercing_hunter_aura = True
            else:
                piercing_hunter = True
        if opt.adds_precision_fighter_aura:
            if tpl.hero:
                precision_fighter_aura = True
            else:
                precision_fighter = True
        if opt.adds_rapid_advance_aura:
            if tpl.hero:
                rapid_advance_aura = True
            else:
                rapid_advance = True
        if opt.adds_rapid_charge_aura:
            if tpl.hero:
                rapid_charge_aura = True
            else:
                rapid_charge = True

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
    # Hero "self-aura" reflection: an aura affects the model itself even when
    # it is unattached. Mirror the aura sources to direct effective flags so a
    # lone hero benefits from its own aura. For non-hero units a banner-style
    # upgrade likewise applies the aura's effect to the unit directly.
    eff_bane_melee = bane_melee_aura or bane_melee_local
    eff_bane_shoot = bane_shoot_aura or bane_shoot_local
    eff_courage = courage_aura or courage_local
    eff_rapid_rush = rapid_rush_aura or rapid_rush_local
    if regeneration_aura:
        regen = True
    if versatile_reach_aura:
        versatile_reach = True
    # ED hero self-aura reflection
    if clan_warrior_boost_aura:
        clan_warrior_boost = True
    if counter_attack_aura:
        counter_attack = True
    if fearless_aura:
        fearless = True
    if ignores_cover_aura:
        ignores_cover = True
    if melee_evasion_aura:
        melee_evasion = True
    if piercing_hunter_aura:
        piercing_hunter = True
    if precision_fighter_aura:
        precision_fighter = True
    if rapid_advance_aura:
        rapid_advance = True
    if rapid_charge_aura:
        rapid_charge = True
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
        faction=tpl.faction,
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
        battleborn=tpl.battleborn,
        strider=strider,
        versatile_attack=versatile_attack,
        versatile_reach=versatile_reach,
        unstoppable_mark=unstoppable_mark,
        bane_melee=eff_bane_melee,
        bane_shoot=eff_bane_shoot,
        courage=eff_courage,
        rapid_rush=eff_rapid_rush,
        bane_melee_aura=bane_melee_aura,
        bane_shoot_aura=bane_shoot_aura,
        courage_aura=courage_aura,
        rapid_rush_aura=rapid_rush_aura,
        regeneration_aura=regeneration_aura,
        versatile_reach_aura=versatile_reach_aura,
        # ED unit-level rules
        clan_warrior=tpl.clan_warrior,
        clan_warrior_boost=clan_warrior_boost,
        piercing_hunter=piercing_hunter,
        melee_evasion=melee_evasion,
        counter_attack=counter_attack,
        unpredictable_fighter=unpredictable_fighter,
        rapid_advance=rapid_advance,
        rapid_charge=rapid_charge,
        bounding=bounding,
        ed_teleport=ed_teleport,
        vengeance=vengeance,
        isr_mark=isr_mark,
        ignores_cover=ignores_cover,
        slow=tpl.slow,
        precision_fighter=precision_fighter,
        # ED aura sources
        clan_warrior_boost_aura=clan_warrior_boost_aura,
        counter_attack_aura=counter_attack_aura,
        fearless_aura=fearless_aura,
        ignores_cover_aura=ignores_cover_aura,
        melee_evasion_aura=melee_evasion_aura,
        piercing_hunter_aura=piercing_hunter_aura,
        precision_fighter_aura=precision_fighter_aura,
        rapid_advance_aura=rapid_advance_aura,
        rapid_charge_aura=rapid_charge_aura,
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
