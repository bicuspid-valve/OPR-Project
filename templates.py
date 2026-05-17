"""Unit templates and weapon constants for High Elf Fleets."""
from __future__ import annotations

from models import Weapon, UpgradeOption, UpgradeSlot, UnitTemplate


# ===================================================================
# WEAPON CONSTANTS
# ===================================================================

W_SHARDGUN = Weapon("Shardgun", 12, 2, crack=True)
W_SHARD_PISTOL = Weapon("Shard Pistol", 12, 1, crack=True)
W_SHARD_CARBINE = Weapon("Shard Carbine", 18, 2, crack=True)
W_MOUNTED_SHARDGUNS = Weapon("Mounted Shardguns", 12, 2, crack=True)
W_SHARD_CANNON = Weapon("Shard Cannon", 24, 3, ap=1, crack=True)
W_MISSILE_LAUNCHER = Weapon("Missile Launcher", 30, 1, ap=2, deadly=3, unstoppable=True)
W_BURST_LASER = Weapon("Burst Laser", 30, 3, ap=2)
W_SHATTER_CANNON = Weapon("Shatter Cannon", 36, 3, ap=1, rending=True)
W_LASER_CANNON = Weapon("Laser Cannon", 36, 1, ap=3, deadly=3)
W_LIGHT_SHARD_CANNON = Weapon("Light Shard Cannon", 24, 2, ap=1, crack=True)
W_SNIPER_RIFLE = Weapon("Sniper Rifle", 30, 1, ap=1, reliable=True, takedown=True)
W_FUSION_RIFLE = Weapon("Fusion Rifle", 12, 1, ap=4, deadly=3)
W_HEAVY_FLAMER = Weapon("Heavy Flamer", 12, 1, ap=1, blast=3, reliable=True)
W_LASER_PISTOL = Weapon("Laser Pistol", 9, 1, ap=2)
W_WEB_SPINNER = Weapon("Web Spinner", 12, 2, ap=4)
W_SURGE_ROCKETS = Weapon("Surge Rockets", 24, 3, ap=1, unstoppable=True)
W_DISTORTION_GUN = Weapon("Distortion Gun", 12, 1, ap=2, blast=3)

# --- Melee weapon constants ---
W_CCW_A1 = Weapon("CCW", 0, 1, melee=True)
W_CCW_A2 = Weapon("CCW", 0, 2, melee=True)
W_ENERGY_SWORD = Weapon("Energy Sword", 0, 2, ap=1, rending=True, melee=True)
W_ENERGY_SWORD_A1 = Weapon("Energy Sword", 0, 1, ap=1, rending=True, melee=True)
W_STINGER_BLADE = Weapon("Stinger Blade", 0, 2, reliable=True, bane=True, melee=True)
W_STOMP_A2 = Weapon("Stomp", 0, 2, ap=1, melee=True)
W_ARTILLERY_CREW = Weapon("Artillery Crew", 0, 3, melee=True)

# --- New melee weapon constants ---
W_KILLING_AXE = Weapon("Killing Axe", 0, 1, ap=2, deadly=3, melee=True)
W_ENERGY_SPEAR = Weapon("Energy Spear", 0, 2, ap=4, melee=True)
W_MASTER_STINGER_BLADE = Weapon("Master Stinger Blade", 0, 3, reliable=True, bane=True, melee=True)
W_GREAT_CLAW = Weapon("Great Claw", 0, 4, rending=True, melee=True)
W_STOMP_A4 = Weapon("Stomp", 0, 4, ap=1, melee=True)
W_STOMP_A6 = Weapon("Stomp", 0, 6, ap=2, melee=True)
W_STOMP_A8 = Weapon("Stomp", 0, 8, ap=2, melee=True)
W_DUAL_TITAN_CLAWS = Weapon("Dual Titan Claws", 0, 24, rending=True, melee=True)
W_FLAME_SWORD = Weapon("Flame Sword", 0, 12, ap=2, rending=True, melee=True)
W_DUAL_ENERGY_SWORDS = Weapon("Dual Energy Swords", 0, 4, ap=1, rending=True, melee=True)
W_ENERGY_SPEAR_A3 = Weapon("Energy Spear", 0, 3, ap=4, melee=True)
W_ENERGY_SWORD_A3 = Weapon("Energy Sword", 0, 3, ap=1, rending=True, melee=True)

# --- Glider weapon constants ---
W_LASER_BLASTER = Weapon("Laser Blaster", 18, 1, ap=2)
W_SHRED_BLASTER_PISTOL = Weapon("Shred Blaster Pistol", 12, 2, rending=True)
W_HEAVY_LASER_BLASTER = Weapon("Heavy Laser Blaster", 18, 3, ap=2)
W_SHRED_BLASTER = Weapon("Shred Blaster", 24, 4, rending=True)

# --- Shifter weapon constants ---
W_SPINNER_CARBINE = Weapon("Spinner Carbine", 18, 2, ap=4)
W_HEAVY_WEB_SPINNER = Weapon("Heavy Web Spinner", 12, 4, ap=4)
W_DUAL_SHIFTER_PICKS = Weapon("Dual Shifter Picks", 0, 2, bane=True, melee=True)

# --- New ranged weapon constants ---
W_RAPID_SHARD_CANNON = Weapon("Rapid Shard Cannon", 24, 6, ap=1, crack=True)
W_PRISM_CANNON = Weapon("Prism Cannon", 36, 2, ap=4, deadly=6)
W_SPINNER_CANNON = Weapon("Spinner Cannon", 30, 4, ap=4, blast=3)
W_MASTER_SHARD_PISTOL = Weapon("Master Shard Pistol", 12, 2, crack=True)
W_RAPID_BURST_LASER = Weapon("Rapid Burst Laser", 30, 6, ap=2)
W_RAPID_SHATTER_CANNON = Weapon("Rapid Shatter Cannon", 36, 6, ap=1, rending=True)
W_GAZE_OF_DOOM = Weapon("Gaze of Doom", 12, 3, ap=4, deadly=3)

# --- Titan melee weapons ---
W_TITAN_AXE = Weapon("Titan Axe", 0, 6, ap=2, deadly=3, melee=True)
W_TITAN_SWORD = Weapon("Titan Sword", 0, 20, ap=1, rending=True, melee=True)
W_TITAN_SPEAR = Weapon("Titan Spear", 0, 16, ap=4, melee=True)
W_TWIN_HEAVY_WRAITH_CANNON = Weapon("Twin Heavy Wraith Cannon", 24, 2, ap=4, deadly=9)
W_SUN_CANNON = Weapon("Sun Cannon", 36, 3, ap=2, blast=6)

# --- Great Elemental melee weapons ---
W_GREAT_AXE = Weapon("Great Axe", 0, 1, ap=2, deadly=6, melee=True)
W_GREAT_SWORD = Weapon("Great Elemental Sword", 0, 6, ap=1, rending=True, melee=True)
W_GREAT_SPEAR = Weapon("Great Spear", 0, 6, ap=4, melee=True)


# ===================================================================
# TEMPLATE BUILDERS
# ===================================================================

def _gun_platform_slot() -> UpgradeSlot:
    """Common gun platform slot for Protectors."""
    return UpgradeSlot("gun_platform", "Replace 1 Shardgun with gun platform", [
        UpgradeOption("shard_cannon", 30, removes_weapon="Shardgun",
                      adds_weapons=[W_SHARD_CANNON]),
        UpgradeOption("missile_launcher", 35, removes_weapon="Shardgun",
                      adds_weapons=[W_MISSILE_LAUNCHER]),
        UpgradeOption("burst_laser", 40, removes_weapon="Shardgun",
                      adds_weapons=[W_BURST_LASER]),
        UpgradeOption("shatter_cannon", 45, removes_weapon="Shardgun",
                      adds_weapons=[W_SHATTER_CANNON]),
        UpgradeOption("laser_cannon", 45, removes_weapon="Shardgun",
                      adds_weapons=[W_LASER_CANNON]),
    ])


def build_unit_templates() -> list[UnitTemplate]:
    templates: list[UnitTemplate] = []

    # --- Protectors [5] 90pts ---
    templates.append(UnitTemplate(
        "protectors", "Protectors", 90, 5, quality=4, defense=5,
        base_weapons=[W_SHARDGUN] * 5 + [W_CCW_A1] * 5,
        upgrade_slots=[_gun_platform_slot()],
    ))

    # --- Strikers [5] 95pts --- Scout
    templates.append(UnitTemplate(
        "strikers", "Strikers", 95, 5, quality=4, defense=5,
        base_weapons=[W_SHARD_PISTOL] * 5 + [W_CCW_A2] * 5,
        scout=True,
        upgrade_slots=[
            UpgradeSlot("specialist", "Replace 1 Shard Pistol", [
                UpgradeOption("flamer", 15, removes_weapon="Shard Pistol",
                              adds_weapons=[Weapon("Flamer", 12, 1, blast=3, reliable=True)]),
                UpgradeOption("fusion_rifle", 20, removes_weapon="Shard Pistol",
                              adds_weapons=[W_FUSION_RIFLE]),
            ]),
            UpgradeSlot("ccw_swap", "Replace 1 CCW", [
                UpgradeOption("energy_sword_ccw", 5, removes_weapon="CCW",
                              removes_count=1, adds_weapons=[W_ENERGY_SWORD]),
            ]),
            UpgradeSlot("hologram_platform", "Hologram Platform", [
                UpgradeOption("hologram_platform", 25, adds_stealth=True),
            ]),
        ],
    ))

    # --- Acolytes [3] 145pts ---
    templates.append(UnitTemplate(
        "acolytes", "Acolytes", 145, 3, quality=4, defense=4,
        base_weapons=[W_SHARD_PISTOL] * 3 + [W_ENERGY_SWORD] * 3,
        upgrade_slots=[
            UpgradeSlot("melee_swap", "Replace any Energy Sword", [
                UpgradeOption("energy_spear", 5, removes_weapon="Energy Sword",
                              removes_count=1, adds_weapons=[W_ENERGY_SPEAR]),
            ]),
            UpgradeSlot("jetbike", "Add Jetbike to all models", [
                UpgradeOption("jetbike", 100,
                              adds_weapons=[W_MOUNTED_SHARDGUNS] * 3,
                              adds_tough=3,
                              applies_to_all=True),
            ]),
        ],
    ))

    # --- Retributors [5] 90pts (melee unit) ---
    templates.append(UnitTemplate(
        "retributors", "Retributors", 90, 5, quality=3, defense=4,
        base_weapons=[W_ENERGY_SWORD_A1] * 5,
        upgrade_slots=[
            UpgradeSlot("carbines", "Shard Carbines for all", [
                UpgradeOption("shard_carbines", 60,
                              adds_weapons=[W_SHARD_CARBINE] * 5,
                              applies_to_all=True),
            ]),
            UpgradeSlot("twin_carbine", "Replace 1 Shard Carbine with Twin", [
                UpgradeOption("twin_shard_carbine", 15, removes_weapon="Shard Carbine",
                              adds_weapons=[Weapon("Twin Shard Carbine", 18, 4, crack=True)],
                              requires="shard_carbines"),
            ]),
            UpgradeSlot("sergeant", "Sergeant weapon swap", [
                UpgradeOption("sgt_pistol", 10, removes_weapon="Shard Carbine",
                              adds_weapons=[W_SHARD_PISTOL],
                              requires="shard_carbines"),
            ]),
            UpgradeSlot("flicker_shield", "Flicker Shield (requires sgt_pistol)", [
                UpgradeOption("flicker_shield", 25,
                              removes_weapon="Shard Pistol",
                              adds_regeneration=True,
                              requires="sgt_pistol"),
            ]),
            UpgradeSlot("psy_marker", "Psy-Marker", [
                UpgradeOption("psy_marker", 25, adds_piercing_spotter=True),
            ]),
        ],
    ))

    # --- Scorchers [3] 130pts ---
    templates.append(UnitTemplate(
        "scorchers", "Scorchers", 130, 3, quality=3, defense=4,
        base_weapons=[W_HEAVY_FLAMER] * 3 + [W_CCW_A1] * 3,
        upgrade_slots=[
            UpgradeSlot("weapon_swap", "Replace Heavy Flamers with Fusion Rifles", [
                UpgradeOption("fusion_rifles_all", 10,
                              removes_weapon="Heavy Flamer", removes_count=3,
                              adds_weapons=[W_FUSION_RIFLE] * 3,
                              applies_to_all=True),
            ]),
            UpgradeSlot("fusion_pike_swap", "Replace one Heavy Flamer with Fusion Pike", [
                UpgradeOption("fusion_pike", 20,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[Weapon("Fusion Pike", 18, 1, ap=4, deadly=3)]),
            ]),
        ],
    ))

    # --- Snipers [3] 135pts --- Scout, Stealth
    templates.append(UnitTemplate(
        "snipers", "Snipers", 135, 3, quality=4, defense=5,
        base_weapons=[W_SNIPER_RIFLE] * 3 + [W_CCW_A1] * 3,
        scout=True, stealth=True,
        upgrade_slots=[UpgradeSlot("psy_marker", "Psy-Marker", [
            UpgradeOption("psy_marker", 25, adds_piercing_spotter=True),
        ])],
    ))

    # --- Gliders [5] 150pts --- Flying
    templates.append(UnitTemplate(
        "gliders", "Gliders", 150, 5, quality=3, defense=4,
        base_weapons=[W_LASER_PISTOL] * 5 + [W_CCW_A2] * 5,
        flying=True,
        upgrade_slots=[
            # "Replace all Laser Pistols and CCWs" → Laser Blaster + CCW(A1)
            UpgradeSlot("laser_blaster_all", "Replace all Laser Pistols and CCWs", [
                UpgradeOption("laser_blaster", 10,
                              removes_weapon="Laser Pistol", removes_count=5,
                              adds_weapons=[W_LASER_BLASTER] * 5,
                              applies_to_all=True,
                              removes_weapon_2="CCW", removes_count_2=5,
                              adds_weapons_2=[W_CCW_A1] * 5),
            ]),
            # "Replace one Laser Pistol and CCW"
            UpgradeSlot("specialist_1", "Replace one Laser Pistol and CCW", [
                UpgradeOption("laser_pistol_energy_sword", 10,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_ENERGY_SWORD]),
                UpgradeOption("shred_blaster_pistol_energy_sword", 15,
                              removes_weapon="Laser Pistol", removes_count=1,
                              adds_weapons=[W_SHRED_BLASTER_PISTOL, W_ENERGY_SWORD],
                              removes_weapon_2="CCW", removes_count_2=1),
            ]),
            # "Replace one Laser Pistol"
            UpgradeSlot("heavy_weapon", "Replace one Laser Pistol", [
                UpgradeOption("heavy_laser_blaster", 35,
                              removes_weapon="Laser Pistol", removes_count=1,
                              adds_weapons=[W_HEAVY_LASER_BLASTER]),
                UpgradeOption("shred_blaster", 50,
                              removes_weapon="Laser Pistol", removes_count=1,
                              adds_weapons=[W_SHRED_BLASTER]),
            ]),
            # "Upgrade one model with Psy-Marker"
            UpgradeSlot("psy_marker", "Psy-Marker", [
                UpgradeOption("psy_marker", 25, adds_piercing_spotter=True),
            ]),
        ],
    ))

    # --- Shifters [5] 245pts --- Teleport
    templates.append(UnitTemplate(
        "shifters", "Shifters", 245, 5, quality=3, defense=4,
        base_weapons=[W_WEB_SPINNER] * 5 + [W_CCW_A1] * 5,
        teleport=True,
        upgrade_slots=[
            # "Replace one Web Spinner"
            UpgradeSlot("ranged_swap", "Replace one Web Spinner", [
                UpgradeOption("spinner_carbine", 15,
                              removes_weapon="Web Spinner", removes_count=1,
                              adds_weapons=[W_SPINNER_CARBINE]),
                UpgradeOption("heavy_web_spinner", 25,
                              removes_weapon="Web Spinner", removes_count=1,
                              adds_weapons=[W_HEAVY_WEB_SPINNER]),
            ]),
            # "Replace one CCW"
            UpgradeSlot("melee_swap", "Replace one CCW", [
                UpgradeOption("dual_shifter_picks", 10,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_DUAL_SHIFTER_PICKS]),
            ]),
        ],
    ))

    # --- Revenants [5] 165pts ---
    templates.append(UnitTemplate(
        "revenants", "Revenants", 165, 5, quality=3, defense=4,
        base_weapons=[W_SHARD_PISTOL] * 5 + [W_ENERGY_SWORD] * 5,
        upgrade_slots=[
            UpgradeSlot("sgt_base", "Replace one Shard Pistol and Energy Sword", [
                UpgradeOption("sgt_base_swap", 0,
                              removes_weapon="Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Sgt. Shard Pistol", 12, 1, crack=True)],
                              removes_weapon_2="Energy Sword", removes_count_2=1,
                              adds_weapons_2=[Weapon("Sgt. Energy Sword", 0, 2, ap=1, rending=True, melee=True)]),
                UpgradeOption("mirror_scythe", 5,
                              removes_weapon="Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Mirror Scythe", 0, 4, rending=True, melee=True)],
                              removes_weapon_2="Energy Sword", removes_count_2=1),
                UpgradeOption("tri_chakram", 20,
                              removes_weapon="Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Tri-Chakram", 12, 3, ap=1, reliable=True)],
                              removes_weapon_2="Energy Sword", removes_count_2=1,
                              adds_weapons_2=[Weapon("Sword", 0, 2, melee=True)]),
            ]),
            UpgradeSlot("sgt_melee_swap", "Replace Sgt. Energy Sword", [
                UpgradeOption("sgt_killing_axe", 5,
                              removes_weapon="Sgt. Energy Sword", removes_count=1,
                              adds_weapons=[W_KILLING_AXE],
                              requires="sgt_base_swap"),
            ]),
            UpgradeSlot("banshee_howl", "Upgrade one model with Banshee Howl", [
                UpgradeOption("banshee_howl", 20, adds_fear=2),
            ]),
        ],
    ))

    # --- Stingers [5] 190pts --- Scout, Stealth
    templates.append(UnitTemplate(
        "stingers", "Stingers", 190, 5, quality=3, defense=4,
        base_weapons=[W_SHARD_PISTOL] * 5 + [W_STINGER_BLADE] * 5,
        scout=True, stealth=True,
        upgrade_slots=[
            UpgradeSlot("sgt_base", "Replace one Shard Pistol and Stinger Blade", [
                UpgradeOption("sgt_base_swap", 0,
                              removes_weapon="Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Sgt. Shard Pistol", 12, 1, crack=True)],
                              removes_weapon_2="Stinger Blade", removes_count_2=1,
                              adds_weapons_2=[Weapon("Sgt. Stinger Blade", 0, 2, reliable=True, bane=True, melee=True)]),
                UpgradeOption("dual_energy_swords", 10,
                              removes_weapon="Shard Pistol", removes_count=1,
                              adds_weapons=[W_DUAL_ENERGY_SWORDS],
                              removes_weapon_2="Stinger Blade", removes_count_2=1),
            ]),
            UpgradeSlot("sgt_ranged_swap", "Replace Sgt. Shard Pistol", [
                UpgradeOption("scorpion_fist", 15,
                              removes_weapon="Sgt. Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Scorpion Fist", 12, 3, rending=True)],
                              requires="sgt_base_swap"),
            ]),
            UpgradeSlot("sgt_melee_swap", "Replace Sgt. Stinger Blade", [
                UpgradeOption("stinger_dagger", 5,
                              removes_weapon="Sgt. Stinger Blade", removes_count=1,
                              adds_weapons=[Weapon("Stinger Dagger", 0, 1, blast=3, reliable=True, bane=True, melee=True)],
                              requires="sgt_base_swap"),
            ]),
        ],
    ))

    # --- Vanquishers [3] 250pts --- Relentless
    templates.append(UnitTemplate(
        "vanquishers", "Vanquishers", 250, 3, quality=3, defense=4,
        base_weapons=[W_SURGE_ROCKETS] * 3 + [W_CCW_A1] * 3,
        relentless=True,
        upgrade_slots=[
            UpgradeSlot("rocket_1", "Replace rocket 1", [
                UpgradeOption("impact_rocket_1", 15, removes_weapon="Surge Rockets",
                              adds_weapons=[Weapon("Impact Rockets", 30, 1, ap=3, deadly=3)]),
                UpgradeOption("shard_cannon_1", 0, removes_weapon="Surge Rockets",
                              adds_weapons=[W_SHARD_CANNON]),
                UpgradeOption("arcing_rockets_1", 20, removes_weapon="Surge Rockets",
                              adds_weapons=[Weapon("Arcing Rockets", 30, 1, ap=1, blast=3)]),
            ]),
            UpgradeSlot("rocket_2", "Replace rocket 2", [
                UpgradeOption("impact_rocket_2", 15, removes_weapon="Surge Rockets",
                              adds_weapons=[Weapon("Impact Rockets", 30, 1, ap=3, deadly=3)]),
            ]),
            UpgradeSlot("rocket_3", "Replace rocket 3", [
                UpgradeOption("impact_rocket_3", 15, removes_weapon="Surge Rockets",
                              adds_weapons=[Weapon("Impact Rockets", 30, 1, ap=3, deadly=3)]),
            ]),
            UpgradeSlot("psy_marker", "Psy-Marker", [
                UpgradeOption("psy_marker", 25, adds_piercing_spotter=True),
            ]),
        ],
    ))

    # --- Elemental Protectors [3] 230pts --- Relentless, Fearless
    templates.append(UnitTemplate(
        "elemental_protectors", "Elemental Protectors", 230, 3,
        quality=3, defense=3, tough=3, fearless=True,
        relentless=True,
        base_weapons=[W_DISTORTION_GUN] * 3 + [W_CCW_A1] * 3,
        upgrade_slots=[UpgradeSlot("weapon_swap", "Replace all Distortion Guns", [
            UpgradeOption("wraith_cannons", 50,
                          removes_weapon="Distortion Gun", removes_count=3,
                          adds_weapons=[Weapon("Wraith Cannon", 12, 3, ap=4)] * 3,
                          applies_to_all=True),
        ])],
    ))

    # --- Jetbike Protectors [3] 200pts --- Fast
    templates.append(UnitTemplate(
        "jetbike_protectors", "Jetbike Protectors", 200, 3,
        quality=3, defense=4, tough=3,
        fast=True,
        base_weapons=[W_MOUNTED_SHARDGUNS] * 3 + [W_SHARD_PISTOL] * 3 + [W_CCW_A2] * 3,
        upgrade_slots=[
            UpgradeSlot("sniper_1", "Add Sniper Rifle to model 1", [
                UpgradeOption("sniper_rifle_1", 30, adds_weapons=[W_SNIPER_RIFLE]),
            ]),
            UpgradeSlot("sniper_2", "Add Sniper Rifle to model 2", [
                UpgradeOption("sniper_rifle_2", 30, adds_weapons=[W_SNIPER_RIFLE]),
            ]),
            UpgradeSlot("sniper_3", "Add Sniper Rifle to model 3", [
                UpgradeOption("sniper_rifle_3", 30, adds_weapons=[W_SNIPER_RIFLE]),
            ]),
            UpgradeSlot("heavy_1", "Replace Shardgun + Sniper on model 1", [
                UpgradeOption("jbp_shard_cannon_1", 5,
                              removes_weapon="Mounted Shardguns",
                              adds_weapons=[W_SHARD_CANNON],
                              requires="sniper_rifle_1"),
                UpgradeOption("jbp_burst_laser_1", 25,
                              removes_weapon="Mounted Shardguns",
                              adds_weapons=[W_BURST_LASER],
                              requires="sniper_rifle_1"),
            ]),
            UpgradeSlot("heavy_2", "Replace Shardgun + Sniper on model 2", [
                UpgradeOption("jbp_shard_cannon_2", 5,
                              removes_weapon="Mounted Shardguns",
                              adds_weapons=[W_SHARD_CANNON],
                              requires="sniper_rifle_2"),
                UpgradeOption("jbp_burst_laser_2", 25,
                              removes_weapon="Mounted Shardguns",
                              adds_weapons=[W_BURST_LASER],
                              requires="sniper_rifle_2"),
            ]),
            UpgradeSlot("heavy_3", "Replace Shardgun + Sniper on model 3", [
                UpgradeOption("jbp_shard_cannon_3", 5,
                              removes_weapon="Mounted Shardguns",
                              adds_weapons=[W_SHARD_CANNON],
                              requires="sniper_rifle_3"),
                UpgradeOption("jbp_burst_laser_3", 25,
                              removes_weapon="Mounted Shardguns",
                              adds_weapons=[W_BURST_LASER],
                              requires="sniper_rifle_3"),
            ]),
            UpgradeSlot("flicker_shield", "Flicker Shield (Regeneration)", [
                UpgradeOption("flicker_shield", 30, adds_regeneration=True),
            ]),
            UpgradeSlot("stealth_aura", "Hologram Field", [
                UpgradeOption("hologram_field", 25, adds_stealth=True),
            ]),
        ],
    ))

    # --- Jetbike Strikers [3] 215pts --- Fast, Impact(1)
    W_ENERGY_LANCE = Weapon("Energy Lance", 0, 2, ap=1, melee=True, thrust=True)
    templates.append(UnitTemplate(
        "jetbike_strikers", "Jetbike Strikers", 215, 3,
        quality=3, defense=4, tough=3,
        fast=True, impact=1,
        base_weapons=[W_MOUNTED_SHARDGUNS] * 3 + [W_SHARD_PISTOL] * 3 + [W_CCW_A2] * 3,
        upgrade_slots=[
            UpgradeSlot("lance_swap_all", "Replace all Shard Pistols and CCWs", [
                UpgradeOption("energy_lance", 25,
                              removes_weapon="Shard Pistol", removes_count=3,
                              adds_weapons=[W_ENERGY_LANCE] * 3,
                              removes_weapon_2="CCW", removes_count_2=3,
                              applies_to_all=True),
            ]),
            UpgradeSlot("lance_swap_1", "Replace one Energy Lance", [
                UpgradeOption("killing_axe_jbs", 5,
                              removes_weapon="Energy Lance", removes_count=1,
                              adds_weapons=[W_KILLING_AXE],
                              requires="energy_lance"),
                UpgradeOption("energy_spear_jbs", 5,
                              removes_weapon="Energy Lance", removes_count=1,
                              adds_weapons=[W_ENERGY_SPEAR],
                              requires="energy_lance"),
                UpgradeOption("energy_sword_jbs", 5,
                              removes_weapon="Energy Lance", removes_count=1,
                              adds_weapons=[W_ENERGY_SWORD_A3],
                              requires="energy_lance"),
            ]),
            UpgradeSlot("stealth_aura", "Hologram Field", [
                UpgradeOption("hologram_field", 25, adds_stealth=True),
            ]),
            UpgradeSlot("master_crafted", "Master-Crafted Engines", [
                UpgradeOption("master_crafted_engines", 20, adds_scout=True),
            ]),
        ],
    ))

    # --- Anti-Gravity APC [1] 165pts --- Fast, Impact(3)
    templates.append(UnitTemplate(
        "ag_apc", "AG APC", 165, 1, quality=3, defense=2, tough=6,
        fast=True, impact=3,
        base_weapons=[W_MOUNTED_SHARDGUNS],
        upgrade_slots=[
            UpgradeSlot("hull_weapon", "Replace Mounted Shardguns", [
                UpgradeOption("light_shard_cannon", 20,
                              removes_weapon="Mounted Shardguns",
                              adds_weapons=[W_LIGHT_SHARD_CANNON]),
            ]),
            UpgradeSlot("turret", "Add 2x Shard Cannons", [
                UpgradeOption("two_shard_cannons", 100,
                              adds_weapons=[W_SHARD_CANNON, W_SHARD_CANNON]),
            ]),
            UpgradeSlot("turret_swap_1", "Replace Shard Cannon 1 (requires turret)", [
                UpgradeOption("apc_missile_1", 10, removes_weapon="Shard Cannon",
                              adds_weapons=[W_MISSILE_LAUNCHER], requires="two_shard_cannons"),
                UpgradeOption("apc_burst_1", 20, removes_weapon="Shard Cannon",
                              adds_weapons=[W_BURST_LASER], requires="two_shard_cannons"),
                UpgradeOption("apc_shatter_1", 20, removes_weapon="Shard Cannon",
                              adds_weapons=[W_SHATTER_CANNON], requires="two_shard_cannons"),
                UpgradeOption("apc_laser_1", 25, removes_weapon="Shard Cannon",
                              adds_weapons=[W_LASER_CANNON], requires="two_shard_cannons"),
            ]),
            UpgradeSlot("hologram_field", "Hologram Field", [
                UpgradeOption("hologram_field", 15, adds_stealth=True),
            ]),
        ],
    ))

    # --- Heavy Jetbike [1] 185pts --- Fast, Impact(3)
    templates.append(UnitTemplate(
        "heavy_jetbike", "Heavy Jetbike", 185, 1, quality=3, defense=2, tough=6,
        fast=True, impact=3,
        base_weapons=[W_SHARD_CANNON, W_MOUNTED_SHARDGUNS],
        upgrade_slots=[
            UpgradeSlot("hull_weapon", "Replace Mounted Shardguns", [
                UpgradeOption("light_shard_cannon", 20,
                              removes_weapon="Mounted Shardguns",
                              adds_weapons=[W_LIGHT_SHARD_CANNON]),
            ]),
            UpgradeSlot("main_weapon", "Replace Shard Cannon", [
                UpgradeOption("hjb_missile", 10, removes_weapon="Shard Cannon",
                              adds_weapons=[W_MISSILE_LAUNCHER]),
                UpgradeOption("hjb_burst", 20, removes_weapon="Shard Cannon",
                              adds_weapons=[W_BURST_LASER]),
                UpgradeOption("hjb_shatter", 20, removes_weapon="Shard Cannon",
                              adds_weapons=[W_SHATTER_CANNON]),
                UpgradeOption("hjb_laser", 25, removes_weapon="Shard Cannon",
                              adds_weapons=[W_LASER_CANNON]),
            ]),
            UpgradeSlot("hologram_field", "Hologram Field", [
                UpgradeOption("hologram_field", 15, adds_stealth=True),
            ]),
        ],
    ))

    # --- Combat Walker [1] 210pts --- Scout
    templates.append(UnitTemplate(
        "combat_walker", "Combat Walker", 210, 1, quality=3, defense=2, tough=6,
        scout=True,
        base_weapons=[Weapon("Rapid Shard Cannon", 24, 6, ap=1, crack=True), W_STOMP_A2],
        upgrade_slots=[UpgradeSlot("main_weapon", "Replace Rapid Shard Cannon", [
            UpgradeOption("rapid_missile", 25, removes_weapon="Rapid Shard Cannon",
                          adds_weapons=[Weapon("Rapid Missile Launcher", 30, 2, ap=2, deadly=3, unstoppable=True)]),
            UpgradeOption("rapid_burst", 35, removes_weapon="Rapid Shard Cannon",
                          adds_weapons=[Weapon("Rapid Burst Laser", 30, 6, ap=2)]),
            UpgradeOption("rapid_shatter", 40, removes_weapon="Rapid Shard Cannon",
                          adds_weapons=[Weapon("Rapid Shatter Cannon", 36, 6, ap=1, rending=True)]),
            UpgradeOption("rapid_laser", 50, removes_weapon="Rapid Shard Cannon",
                          adds_weapons=[Weapon("Rapid Laser Cannon", 36, 2, ap=3, deadly=3)]),
        ]),
        UpgradeSlot("hologram_field", "Hologram Field", [
            UpgradeOption("hologram_field", 15, adds_stealth=True),
        ])],
    ))

    # --- Support Artillery [1] 165pts --- Artillery
    templates.append(UnitTemplate(
        "support_artillery", "Support Artillery", 165, 1,
        quality=3, defense=3, tough=3,
        artillery=True,
        base_weapons=[Weapon("Burst Mortar", 30, 2, blast=3), W_ARTILLERY_CREW],
        upgrade_slots=[UpgradeSlot("main_weapon", "Replace Burst Mortar", [
            UpgradeOption("heavy_mortar", 20, removes_weapon="Burst Mortar",
                          adds_weapons=[Weapon("Heavy Mortar", 24, 1, ap=2, deadly=6)]),
            UpgradeOption("aa_cannon", 45, removes_weapon="Burst Mortar",
                          adds_weapons=[Weapon("AA-Cannon", 36, 6, ap=1, unstoppable=True)]),
        ]),
        UpgradeSlot("hologram_field", "Hologram Field", [
            UpgradeOption("hologram_field", 5, adds_stealth=True),
        ])],
    ))

    # --- Elven Noble [1] 40pts --- Hero, Shielded
    templates.append(UnitTemplate(
        "elven_noble", "Elven Noble", 40, 1, quality=3, defense=4, tough=3,
        hero=True, shielded=True,
        base_weapons=[W_CCW_A2],
        upgrade_slots=[
            UpgradeSlot("role", "Role upgrade", [
                UpgradeOption("high_avenger", 15),
                UpgradeOption("ancient_commander", 15),
                UpgradeOption("elemental_hexer", 25),
                UpgradeOption("guardian_cleric", 30),
                UpgradeOption("high_oracle", 50),
                UpgradeOption("master_high_oracle", 80),
            ]),
            UpgradeSlot("ranged_weapon", "Ranged weapon", [
                UpgradeOption("master_shard_pistol", 5,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_MASTER_SHARD_PISTOL, W_CCW_A2],
                              removes_shielded=True),
                UpgradeOption("surge_rockets", 40,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_SURGE_ROCKETS],
                              removes_shielded=True),
                UpgradeOption("impact_rockets", 55,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[Weapon("Impact Rockets", 30, 1, ap=3, deadly=3)],
                              removes_shielded=True),
                UpgradeOption("master_sniper_rifle", 50,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[Weapon("Master Sniper Rifle", 30, 2, ap=1, reliable=True, takedown=True)],
                              removes_shielded=True),
            ]),
            UpgradeSlot("replace_msp", "Replace Master Shard Pistol", [
                UpgradeOption("master_laser_pistol", 5,
                              removes_weapon="Master Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Master Laser Pistol", 9, 2, ap=2)],
                              requires="master_shard_pistol", removes_shielded=True),
                UpgradeOption("master_shard_carbine", 15,
                              removes_weapon="Master Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Master Shard Carbine", 18, 3, crack=True)],
                              requires="master_shard_pistol", removes_shielded=True),
                UpgradeOption("master_laser_blaster", 20,
                              removes_weapon="Master Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Master Laser Blaster", 18, 2, ap=2)],
                              requires="master_shard_pistol", removes_shielded=True),
                UpgradeOption("great_flamer_axe", 20,
                              removes_weapon="Master Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Great Flamer Axe", 12, 1, ap=1, blast=3, reliable=True)],
                              requires="master_shard_pistol", removes_shielded=True),
                UpgradeOption("great_fusion_axe", 20,
                              removes_weapon="Master Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Great Fusion Axe", 12, 1, ap=4, deadly=3)],
                              requires="master_shard_pistol", removes_shielded=True),
                UpgradeOption("master_web_spinner", 30,
                              removes_weapon="Master Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Master Web Spinner", 12, 3, ap=4)],
                              requires="master_shard_pistol", removes_shielded=True),
            ]),
            UpgradeSlot("melee_weapon", "Melee weapon", [
                UpgradeOption("energy_sword", 10,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_ENERGY_SWORD]),
                UpgradeOption("killing_axe", 15,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_KILLING_AXE]),
                UpgradeOption("energy_spear", 15,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_ENERGY_SPEAR]),
                UpgradeOption("master_stinger_blade", 20,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_MASTER_STINGER_BLADE]),
                UpgradeOption("dire_sword", 5,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[Weapon("Dire Sword", 0, 1, blast=3, melee=True)]),
            ]),
            UpgradeSlot("shield_projector", "Shield Projector", [
                UpgradeOption("shield_projector", 25, adds_stealth_aura=True),
            ]),
            UpgradeSlot("leader_type", "Leader type", [
                UpgradeOption("vanquisher_leader", 0, adds_relentless=True),
                UpgradeOption("stinger_leader", 15,
                              adds_scout=True, adds_stealth=True),
                UpgradeOption("glider_leader", 15, adds_flying=True),
                UpgradeOption("shifter_leader", 30, adds_teleport=True),
                UpgradeOption("revenant_leader", 25, adds_fear=2),
                UpgradeOption("jetbike_leader", 70,
                              adds_weapons=[W_MOUNTED_SHARDGUNS],
                              adds_tough=3, adds_fast=True),
            ]),
        ],
    ))

    # --- Elite Protector [1] 30pts --- Hero, Shielded
    templates.append(UnitTemplate(
        "elite_protector", "Elite Protector", 30, 1, quality=4, defense=5, tough=3,
        hero=True, shielded=True,
        base_weapons=[W_CCW_A2],
        upgrade_slots=[
            UpgradeSlot("role", "Role upgrade", [
                UpgradeOption("acolyte_psy_seer", 10),
                UpgradeOption("high_avenger", 15),
                UpgradeOption("psy_artillerist", 15),
                UpgradeOption("elemental_hexer", 25),
                UpgradeOption("guardian_cleric", 30),
                UpgradeOption("psy_weaver", 35),
                UpgradeOption("high_oracle", 50),
            ]),
            UpgradeSlot("ranged_weapon", "Ranged weapon", [
                UpgradeOption("master_shard_pistol", 5,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_MASTER_SHARD_PISTOL, W_CCW_A2],
                              removes_shielded=True),
                UpgradeOption("surge_rockets", 30,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_SURGE_ROCKETS],
                              removes_shielded=True),
                UpgradeOption("impact_rockets", 40,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[Weapon("Impact Rockets", 30, 1, ap=3, deadly=3)],
                              removes_shielded=True),
                UpgradeOption("master_sniper_rifle", 55,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[Weapon("Master Sniper Rifle", 30, 2, ap=1, reliable=True, takedown=True)],
                              removes_shielded=True),
            ]),
            UpgradeSlot("replace_msp", "Replace Master Shard Pistol", [
                UpgradeOption("master_laser_pistol", 5,
                              removes_weapon="Master Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Master Laser Pistol", 9, 2, ap=2)],
                              requires="master_shard_pistol", removes_shielded=True),
                UpgradeOption("master_shard_carbine", 10,
                              removes_weapon="Master Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Master Shard Carbine", 18, 3, crack=True)],
                              requires="master_shard_pistol", removes_shielded=True),
                UpgradeOption("master_laser_blaster", 15,
                              removes_weapon="Master Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Master Laser Blaster", 18, 2, ap=2)],
                              requires="master_shard_pistol", removes_shielded=True),
                UpgradeOption("great_flamer_axe", 20,
                              removes_weapon="Master Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Great Flamer Axe", 12, 1, ap=1, blast=3, reliable=True)],
                              requires="master_shard_pistol", removes_shielded=True),
                UpgradeOption("great_fusion_axe", 15,
                              removes_weapon="Master Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Great Fusion Axe", 12, 1, ap=4, deadly=3)],
                              requires="master_shard_pistol", removes_shielded=True),
                UpgradeOption("master_web_spinner", 20,
                              removes_weapon="Master Shard Pistol", removes_count=1,
                              adds_weapons=[Weapon("Master Web Spinner", 12, 3, ap=4)],
                              requires="master_shard_pistol", removes_shielded=True),
            ]),
            UpgradeSlot("melee_weapon", "Melee weapon", [
                UpgradeOption("energy_sword", 5,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_ENERGY_SWORD]),
                UpgradeOption("killing_axe", 10,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_KILLING_AXE]),
                UpgradeOption("energy_spear", 15,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_ENERGY_SPEAR]),
                UpgradeOption("master_stinger_blade", 20,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[W_MASTER_STINGER_BLADE]),
                UpgradeOption("dire_sword", 5,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[Weapon("Dire Sword", 0, 1, blast=3, melee=True)]),
            ]),
            UpgradeSlot("shield_projector", "Shield Projector", [
                UpgradeOption("shield_projector", 25, adds_stealth_aura=True),
            ]),
            UpgradeSlot("leader_type", "Leader type", [
                UpgradeOption("striker_leader", 5, adds_scout=True),
                UpgradeOption("sniper_leader", 15,
                              adds_scout=True, adds_stealth=True),
                UpgradeOption("jetbike_leader", 50,
                              adds_weapons=[W_MOUNTED_SHARDGUNS],
                              adds_tough=3, adds_fast=True),
            ]),
        ],
    ))

    # --- Elemental Strikers [3] 235pts --- Fearless, Furious, Shielded
    templates.append(UnitTemplate(
        "elemental_strikers", "Elemental Strikers", 235, 3,
        quality=3, defense=3, tough=3,
        fearless=True, furious=True, shielded=True,
        base_weapons=[W_KILLING_AXE] * 3,
        upgrade_slots=[
            UpgradeSlot("weapon_swap_all", "Replace all Killing Axes", [
                UpgradeOption("dual_energy_swords", 30,
                              removes_weapon="Killing Axe", removes_count=3,
                              adds_weapons=[W_DUAL_ENERGY_SWORDS] * 3,
                              applies_to_all=True, removes_shielded=True),
            ]),
            UpgradeSlot("weapon_swap_1", "Replace 1 Killing Axe", [
                UpgradeOption("energy_sword_1", 5,
                              removes_weapon="Killing Axe", removes_count=1,
                              adds_weapons=[W_ENERGY_SWORD_A3]),
                UpgradeOption("energy_spear_1", 20,
                              removes_weapon="Killing Axe", removes_count=1,
                              adds_weapons=[W_ENERGY_SPEAR_A3]),
            ]),
        ],
    ))

    # --- Anti-Gravity Tank [1] 465pts --- Fast, Impact(6)
    templates.append(UnitTemplate(
        "ag_tank", "AG Tank", 465, 1, quality=3, defense=2, tough=12,
        fast=True, impact=6,
        base_weapons=[W_RAPID_SHARD_CANNON, W_RAPID_SHARD_CANNON, W_MOUNTED_SHARDGUNS],
        upgrade_slots=[
            UpgradeSlot("main_weapon_all", "Replace both Rapid Shard Cannons", [
                UpgradeOption("prism_cannon", 70,
                              removes_weapon="Rapid Shard Cannon", removes_count=2,
                              adds_weapons=[W_PRISM_CANNON]),
                UpgradeOption("spinner_cannon", 290,
                              removes_weapon="Rapid Shard Cannon", removes_count=2,
                              adds_weapons=[W_SPINNER_CANNON]),
            ]),
            UpgradeSlot("main_weapon_1", "Replace 1 Rapid Shard Cannon", [
                UpgradeOption("rapid_missile_1", 25, removes_weapon="Rapid Shard Cannon",
                              adds_weapons=[Weapon("Rapid Missile Launcher", 30, 2, ap=2, deadly=3, unstoppable=True)]),
                UpgradeOption("rapid_burst_1", 35, removes_weapon="Rapid Shard Cannon",
                              adds_weapons=[W_RAPID_BURST_LASER]),
                UpgradeOption("rapid_shatter_1", 40, removes_weapon="Rapid Shard Cannon",
                              adds_weapons=[W_RAPID_SHATTER_CANNON]),
                UpgradeOption("rapid_laser_1", 50, removes_weapon="Rapid Shard Cannon",
                              adds_weapons=[Weapon("Rapid Laser Cannon", 36, 2, ap=3, deadly=3)]),
            ]),
            UpgradeSlot("hull_weapon", "Replace Mounted Shardguns", [
                UpgradeOption("light_shard_cannon", 20,
                              removes_weapon="Mounted Shardguns",
                              adds_weapons=[W_LIGHT_SHARD_CANNON]),
            ]),
            UpgradeSlot("hologram_field", "Hologram Field", [
                UpgradeOption("hologram_field", 25, adds_stealth=True),
            ]),
        ],
    ))

    # --- Great Elemental [1] 340pts --- Fearless, Fear(2)
    templates.append(UnitTemplate(
        "great_elemental", "Great Elemental", 340, 1, quality=3, defense=2, tough=12,
        fearless=True, fear=2,
        base_weapons=[W_SHARDGUN, W_SHARDGUN, W_GREAT_CLAW, W_GREAT_CLAW, W_STOMP_A4],
        upgrade_slots=[
            UpgradeSlot("melee_swap", "Replace 2x Great Claw", [
                UpgradeOption("great_axe_shield", 0,
                              removes_weapon="Great Claw", removes_count=2,
                              adds_weapons=[W_GREAT_AXE],
                              adds_fortified=True),
                UpgradeOption("great_sword_shield", 15,
                              removes_weapon="Great Claw", removes_count=2,
                              adds_weapons=[W_GREAT_SWORD],
                              adds_fortified=True),
                UpgradeOption("dual_great_axe", 25,
                              removes_weapon="Great Claw", removes_count=2,
                              adds_weapons=[W_GREAT_AXE, W_GREAT_AXE]),
                UpgradeOption("great_spear_shield", 40,
                              removes_weapon="Great Claw", removes_count=2,
                              adds_weapons=[W_GREAT_SPEAR],
                              adds_fortified=True),
                UpgradeOption("dual_great_sword", 55,
                              removes_weapon="Great Claw", removes_count=2,
                              adds_weapons=[W_GREAT_SWORD, W_GREAT_SWORD]),
            ]),
            UpgradeSlot("ranged_swap", "Replace 1 Shardgun", [
                UpgradeOption("flamer_ge", 10,
                              removes_weapon="Shardgun", removes_count=1,
                              adds_weapons=[Weapon("Flamer", 12, 1, blast=3, reliable=True)]),
            ]),
            UpgradeSlot("heavy_ranged", "Add heavy ranged weapon", [
                UpgradeOption("rapid_shard_cannon_ge", 100,
                              adds_weapons=[W_RAPID_SHARD_CANNON]),
                UpgradeOption("rapid_burst_laser_ge", 135,
                              adds_weapons=[W_RAPID_BURST_LASER]),
                UpgradeOption("rapid_shatter_cannon_ge", 140,
                              adds_weapons=[W_RAPID_SHATTER_CANNON]),
                UpgradeOption("rapid_missile_launcher_ge", 120,
                              adds_weapons=[Weapon("Rapid Missile Launcher", 30, 2, ap=2, deadly=3, unstoppable=True)]),
                UpgradeOption("rapid_laser_cannon_ge", 150,
                              adds_weapons=[Weapon("Rapid Laser Cannon", 36, 2, ap=3, deadly=3)]),
            ]),
        ],
    ))

    # --- Titan Elemental [1] 755pts --- Fearless, Fear(4)
    templates.append(UnitTemplate(
        "titan_elemental", "Titan Elemental", 755, 1, quality=3, defense=2, tough=24,
        fearless=True, fear=4,
        base_weapons=[W_DUAL_TITAN_CLAWS, W_STOMP_A8],
        upgrade_slots=[
            UpgradeSlot("melee_swap", "Replace Dual Titan Claws", [
                UpgradeOption("titan_axe_shield", 15,
                              removes_weapon="Dual Titan Claws",
                              adds_weapons=[W_TITAN_AXE],
                              adds_fortified=True),
                UpgradeOption("twin_heavy_wraith_cannon", 140,
                              removes_weapon="Dual Titan Claws",
                              adds_weapons=[W_TWIN_HEAVY_WRAITH_CANNON]),
            ]),
            UpgradeSlot("axe_swap", "Replace Titan Axe (requires titan_axe_shield)", [
                UpgradeOption("titan_sword", 35,
                              removes_weapon="Titan Axe",
                              adds_weapons=[W_TITAN_SWORD],
                              requires="titan_axe_shield"),
                UpgradeOption("titan_spear", 65,
                              removes_weapon="Titan Axe",
                              adds_weapons=[W_TITAN_SPEAR],
                              requires="titan_axe_shield"),
                UpgradeOption("sun_cannon", 170,
                              removes_weapon="Titan Axe",
                              adds_weapons=[W_SUN_CANNON],
                              requires="titan_axe_shield"),
            ]),
            UpgradeSlot("ranged_1", "Add ranged weapon 1", [
                UpgradeOption("rapid_shard_cannon_t1", 100,
                              adds_weapons=[W_RAPID_SHARD_CANNON]),
                UpgradeOption("rapid_burst_laser_t1", 135,
                              adds_weapons=[W_RAPID_BURST_LASER]),
                UpgradeOption("rapid_shatter_cannon_t1", 140,
                              adds_weapons=[W_RAPID_SHATTER_CANNON]),
            ]),
            UpgradeSlot("ranged_2", "Add ranged weapon 2", [
                UpgradeOption("rapid_shard_cannon_t2", 100,
                              adds_weapons=[W_RAPID_SHARD_CANNON]),
                UpgradeOption("rapid_burst_laser_t2", 135,
                              adds_weapons=[W_RAPID_BURST_LASER]),
                UpgradeOption("rapid_shatter_cannon_t2", 140,
                              adds_weapons=[W_RAPID_SHATTER_CANNON]),
            ]),
        ],
    ))

    # --- Elemental Avatar [1] 870pts --- Fearless, Fear(3), Regeneration
    templates.append(UnitTemplate(
        "elemental_avatar", "Elemental Avatar", 870, 1, quality=2, defense=2, tough=18,
        fearless=True, fear=3, regeneration=True,
        base_weapons=[W_FLAME_SWORD, W_STOMP_A6],
        upgrade_slots=[
            UpgradeSlot("avatar_type", "Great Avatar upgrade", [
                UpgradeOption("great_avatar", 45,
                              adds_weapons=[W_GAZE_OF_DOOM]),
            ]),
        ],
    ))

    # Tag every HEF template with its faction. Doing it once here keeps the
    # individual UnitTemplate(...) call sites free of boilerplate.
    for tpl in templates:
        tpl.faction = "hef"
    return templates


# ===================================================================
# BATTLE BROTHERS — WEAPON CONSTANTS
# ===================================================================
# Naming: BB_W_<NAME>. Where a BB weapon's stat-line matches an existing HEF
# constant we just reuse the HEF symbol below.

# Pistols & rifles
BB_W_HEAVY_PISTOL = Weapon("Heavy Pistol", 12, 1, ap=1)
BB_W_SGT_HEAVY_PISTOL = Weapon("Sgt. Heavy Pistol", 12, 1, ap=1)
BB_W_MASTER_HEAVY_PISTOL = Weapon("Master Heavy Pistol", 12, 2, ap=1)
BB_W_FLAMER_PISTOL = Weapon("Flamer Pistol", 6, 1, blast=3, reliable=True)
BB_W_PLASMA_PISTOL = Weapon("Plasma Pistol", 12, 1, ap=4)
BB_W_MASTER_PLASMA_PISTOL = Weapon("Master Plasma Pistol", 12, 2, ap=4)
BB_W_FUSION_PISTOL = Weapon("Fusion Pistol", 6, 1, ap=4, deadly=3)
BB_W_GRAVITY_PISTOL = Weapon("Gravity Pistol", 9, 1, ap=1, blast=3, lacerate=True)
BB_W_HEAVY_RIFLE = Weapon("Heavy Rifle", 24, 1, ap=1)
BB_W_MASTER_HEAVY_RIFLE = Weapon("Master Heavy Rifle", 24, 2, ap=1)
BB_W_TWIN_HEAVY_RIFLE = Weapon("Twin Heavy Rifle", 24, 2, ap=1)
BB_W_STORM_RIFLE_A3 = Weapon("Storm Rifle", 24, 3, ap=1)
BB_W_STORM_RIFLE_A4 = Weapon("Storm Rifle", 24, 4, ap=1)  # Master Destroyer base
BB_W_STORM_RIFLE_A6 = Weapon("Storm Rifle", 24, 6, ap=1)  # APC variant
BB_W_MASTER_STORM_RIFLE = Weapon("Master Storm Rifle", 24, 4, ap=1)
BB_W_RAPID_STORM_RIFLE = Weapon("Rapid Storm Rifle", 24, 6, ap=1)
BB_W_SHOTGUN = Weapon("Shotgun", 12, 2, ap=1)
BB_W_MASTER_SHOTGUN = Weapon("Master Shotgun", 12, 3, ap=1)
BB_W_SNIPER_RIFLE = Weapon("Sniper Rifle", 30, 1, ap=1, reliable=True, takedown=True)
BB_W_MASTER_SNIPER_RIFLE = Weapon("Master Sniper Rifle", 30, 2, ap=1, reliable=True, takedown=True)

# Heavy / special / cannon-class weapons
BB_W_FLAMER = Weapon("Flamer", 12, 1, blast=3, reliable=True)
BB_W_HEAVY_FLAMER = W_HEAVY_FLAMER  # 12, A1, AP(1), Blast(3), Reliable
BB_W_TWIN_HEAVY_FLAMER = Weapon("Twin Heavy Flamer", 12, 2, ap=1, blast=3, reliable=True)
BB_W_TWIN_FLAMER = Weapon("Twin Flamer", 12, 2, blast=3, reliable=True)
BB_W_QUAD_FLAMER_CANNON = Weapon("Quad Flamer Cannon", 18, 4, ap=1, blast=3, reliable=True)
BB_W_DEATH_LAUNCHER = Weapon("Death Launcher", 18, 1, blast=6)
BB_W_GRENADE_LAUNCHER = Weapon("Grenade Launcher", 24, 1, blast=3)
BB_W_FUSION_RIFLE = W_FUSION_RIFLE  # matches HEF Fusion Rifle
BB_W_HEAVY_FUSION_RIFLE = Weapon("Heavy Fusion Rifle", 18, 1, ap=4, deadly=3)
BB_W_TWIN_FUSION_RIFLE = Weapon("Twin Fusion Rifle", 12, 2, ap=4, deadly=3)
BB_W_SUPER_HEAVY_FUSION_RIFLE = Weapon("Super-Heavy Fusion Rifle", 18, 1, ap=4, deadly=6)
BB_W_PLASMA_RIFLE = Weapon("Plasma Rifle", 24, 1, ap=4)
BB_W_PLASMA_CANNON = Weapon("Plasma Cannon", 30, 1, ap=4, blast=3)
BB_W_HEAVY_PLASMA_CANNON = Weapon("Heavy Plasma Cannon", 30, 1, ap=4, blast=6)
BB_W_TWIN_PLASMA_CANNON = Weapon("Twin Plasma Cannon", 30, 2, ap=4, blast=3)
BB_W_GRAVITY_RIFLE = Weapon("Gravity Rifle", 18, 1, ap=1, smash=True)
BB_W_GRAVITY_CANNON = Weapon("Gravity Cannon", 24, 1, ap=2, smash=True)
BB_W_TWIN_LIGHT_GRAVITY_CANNON = Weapon("Twin Light Gravity Cannon", 24, 2, ap=2, smash=True)
BB_W_HEAVY_MACHINEGUN = Weapon("Heavy Machinegun", 30, 3, ap=1)
BB_W_TWIN_HEAVY_MACHINEGUN = Weapon("Twin Heavy Machinegun", 30, 6, ap=1)
BB_W_HEAVY_RIFLE_ARRAY = Weapon("Heavy Rifle Array", 24, 6, ap=1)
BB_W_TWIN_HEAVY_RIFLE_ARRAY = Weapon("Twin Heavy Rifle Array", 24, 12, ap=1)
BB_W_LIGHT_HEAVY_RIFLE_ARRAY = Weapon("Light Heavy Rifle Array", 24, 4, ap=1)
BB_W_MINIGUN = Weapon("Minigun", 24, 4, ap=1)
BB_W_TWIN_MINIGUN = Weapon("Twin Minigun", 24, 8, ap=1)
BB_W_HEAVY_MINIGUN = Weapon("Heavy Minigun", 24, 6, ap=2)
BB_W_HEAVY_GATLING_CANNON = Weapon("Heavy Gatling Cannon", 24, 6, ap=1)
BB_W_HEAVY_FLAK_CANNON = Weapon("Heavy Flak Cannon", 30, 2, ap=3, deadly=3, unstoppable=True)
BB_W_HEAVY_THUNDER_CANNON = Weapon("Heavy Thunder Cannon", 30, 2, ap=2, blast=3, indirect=True)
BB_W_HEAVY_CRACK_CANNON = Weapon("Heavy Crack Cannon", 30, 6, ap=1, indirect=True, rending=True)
BB_W_DEMOLITION_CANNON = Weapon("Demolition Cannon", 24, 1, ap=4, blast=6, indirect=True)
BB_W_WIND_MISSILE_LAUNCHER = Weapon("Wind Missile Launcher", 30, 2, ap=1, blast=3, indirect=True)
BB_W_LASER_CANNON = W_LASER_CANNON  # matches HEF
BB_W_TWIN_LASER_CANNON = Weapon("Twin Laser Cannon", 36, 2, ap=3, deadly=3)
BB_W_QUAD_LASER_CANNON = Weapon("Quad Laser Cannon", 36, 4, ap=3, deadly=3)
BB_W_LASER_TALON = Weapon("Laser Talon", 24, 2, ap=3)
BB_W_RAPID_AUTOCANNON = Weapon("Rapid Autocannon", 36, 6, ap=2)
BB_W_TWIN_AUTOCANNON = Weapon("Twin Autocannon", 36, 6, ap=2)

# Missiles / unstoppable launchers
BB_W_MISSILE_LAUNCHER = Weapon("Missile Launcher", 30, 1, ap=2, deadly=3, unstoppable=True)
BB_W_MISSILE_ARRAY = Weapon("Missile Array", 30, 4, ap=2, unstoppable=True)
BB_W_TWIN_TYPHOON_MISSILES = Weapon("Twin Typhoon Missiles", 24, 4, ap=2, unstoppable=True)
BB_W_CYCLONE_MISSILES = Weapon("Cyclone Missiles", 24, 1, ap=2, deadly=3, unstoppable=True)
BB_W_SPEAR_MISSILE_LAUNCHER = Weapon("Spear Missile Launcher", 30, 1, ap=3, deadly=6, unstoppable=True)
BB_W_TWIN_HAMMER_MISSILES = Weapon("Twin Hammer Missiles", 36, 2, ap=3, deadly=3, unstoppable=True)
BB_W_HUNTER_MISSILES = Weapon("Hunter Missiles", 24, 1, ap=2, deadly=3, limited=True, unstoppable=True)
BB_W_CHEST_MISSILES = Weapon("Chest Missiles", 24, 1, ap=2, unstoppable=True)
BB_W_CHEST_RIFLES = Weapon("Chest-Rifles", 24, 2, ap=1)

# Master Heavy Rifle attachments — all Limited
BB_W_FLAMER_MOD = Weapon("Flamer-Mod", 12, 1, blast=3, limited=True, reliable=True)
BB_W_PLASMA_MOD = Weapon("Plasma-Mod", 24, 1, ap=4, limited=True)
BB_W_FUSION_MOD = Weapon("Fusion-Mod", 12, 1, ap=4, deadly=3, limited=True)
BB_W_GRAVITY_MOD = Weapon("Gravity-Mod", 18, 1, ap=1, blast=3, limited=True, lacerate=True)

# Melee
BB_W_CCW_A1 = W_CCW_A1
BB_W_CCW_A2 = W_CCW_A2
BB_W_CCW_A3 = Weapon("CCW", 0, 3, melee=True)
BB_W_CCW_A4 = Weapon("CCW", 0, 4, melee=True)
BB_W_BASH_A1 = Weapon("Bash", 0, 1, melee=True)
BB_W_HEAVY_CCW = Weapon("Heavy CCW", 0, 2, ap=1, melee=True)
BB_W_DUAL_ENERGY_CLAWS_A4 = Weapon("Dual Energy Claws", 0, 4, rending=True, melee=True)
BB_W_DUAL_ENERGY_CLAWS_A8 = Weapon("Dual Energy Claws", 0, 8, rending=True, melee=True)
BB_W_HEAVY_CHAINSAW = Weapon("Heavy Chainsaw Sword", 0, 4, ap=1, melee=True)
BB_W_ENERGY_HAMMER_A1 = Weapon("Energy Hammer", 0, 1, blast=3, melee=True)
BB_W_ENERGY_HAMMER_A2 = Weapon("Energy Hammer", 0, 2, blast=3, melee=True)
BB_W_ENERGY_HAMMER_A3 = Weapon("Energy Hammer", 0, 3, blast=3, melee=True)
BB_W_ENERGY_SWORD_A1 = Weapon("Energy Sword", 0, 1, ap=1, rending=True, melee=True)
BB_W_ENERGY_SWORD_A2 = Weapon("Energy Sword", 0, 2, ap=1, rending=True, melee=True)
BB_W_ENERGY_SWORD_A3 = Weapon("Energy Sword", 0, 3, ap=1, rending=True, melee=True)
BB_W_ENERGY_SWORD_A4 = Weapon("Energy Sword", 0, 4, ap=1, rending=True, melee=True)
BB_W_ENERGY_FIST_A2 = Weapon("Energy Fist", 0, 2, ap=4, melee=True)
BB_W_ENERGY_FIST_A3 = Weapon("Energy Fist", 0, 3, ap=4, melee=True)
BB_W_CHAIN_FIST_A1 = Weapon("Chain-Fist", 0, 1, ap=2, deadly=3, melee=True)
BB_W_CHAIN_FIST_A2 = Weapon("Chain-Fist", 0, 2, ap=2, deadly=3, melee=True)
BB_W_CHAIN_FIST_A3 = Weapon("Chain-Fist", 0, 3, ap=2, deadly=3, melee=True)
BB_W_DUAL_HEAVY_FISTS = Weapon("Dual Heavy Fists", 0, 2, blast=3, melee=True)
BB_W_DUAL_COMBAT_DRILLS = Weapon("Dual Combat Drills", 0, 4, ap=4, melee=True)
BB_W_STOMP_A2 = Weapon("Stomp", 0, 2, ap=1, melee=True)
BB_W_STOMP_A4 = Weapon("Stomp", 0, 4, ap=1, melee=True)
BB_W_STOMP_A6 = Weapon("Stomp", 0, 6, ap=1, melee=True)
BB_W_WALKER_FIST = Weapon("Walker Fist", 0, 4, ap=4, melee=True)
BB_W_TWIN_STORM_CANNON = Weapon("Twin Storm Cannon", 30, 4, ap=2, unstoppable=True)


# ===================================================================
# BATTLE BROTHERS — UNIT TEMPLATES
# ===================================================================
# Implements the GF Battle Brothers v3.5.3 list with the user's adjustments:
#   - Drop Pod, Light Gunship and Heavy Gunship are not in the roster
#   - Ambush is stripped from the units that would otherwise have it
#   - Transport(N) is ignored (units stay; the rule does nothing in sim)
#   - Caster, Mend, Re-Deployment, Re-Position Artillery and Melee Shrouding
#     upgrade options are dropped
#   - Versatile Reach is modelled as permanent +4" range and +2" charge
#   - Limited weapons fire on the unit's first activation, then are stripped

def build_bb_unit_templates() -> list[UnitTemplate]:
    templates: list[UnitTemplate] = []

    # ---------- HEROES ----------

    # Master Destroyer [1] — 140pts (Ambush dropped per user)
    templates.append(UnitTemplate(
        "bb_master_destroyer", "Master Destroyer", 140, 1, quality=3, defense=3,
        tough=6, fearless=True, hero=True, shielded=True, faction="bb",
        battleborn=True,
        base_weapons=[BB_W_CCW_A4],
        upgrade_slots=[
            UpgradeSlot("aura_command", "Command aura (one)", [
                UpgradeOption("lieutenant", 15, adds_rapid_rush_aura=True),
                UpgradeOption("preacher", 20, adds_bane_melee_aura=True),
                UpgradeOption("captain", 30, adds_bane_shoot_aura=True),
            ]),
            UpgradeSlot("ccw_swap", "Replace Combat Shield and CCW", [
                UpgradeOption("dual_energy_claws", 20, removes_weapon="CCW",
                              adds_weapons=[BB_W_DUAL_ENERGY_CLAWS_A8],
                              removes_shielded=True),
                UpgradeOption("rapid_storm_rifle", 70, removes_weapon="CCW",
                              adds_weapons=[BB_W_RAPID_STORM_RIFLE, BB_W_CCW_A4],
                              removes_shielded=True),
            ]),
            UpgradeSlot("melee_swap", "Replace CCW", [
                UpgradeOption("energy_hammer", 5, removes_weapon="CCW",
                              adds_weapons=[BB_W_ENERGY_HAMMER_A2]),
                UpgradeOption("energy_sword", 15, removes_weapon="CCW",
                              adds_weapons=[BB_W_ENERGY_SWORD_A4]),
                UpgradeOption("chain_fist", 25, removes_weapon="CCW",
                              adds_weapons=[BB_W_CHAIN_FIST_A2]),
                UpgradeOption("energy_fist", 30, removes_weapon="CCW",
                              adds_weapons=[BB_W_ENERGY_FIST_A3]),
            ]),
        ],
    ))

    # Veteran Master Brother [1] — 60pts (Versatile Attack)
    templates.append(UnitTemplate(
        "bb_vet_master_brother", "Veteran Master Brother", 60, 1, quality=3, defense=3,
        tough=3, fearless=True, hero=True, shielded=True, faction="bb",
        battleborn=True, versatile_attack=True,
        base_weapons=[BB_W_CCW_A2],
        upgrade_slots=[
            # Caster, Mend, Re-Deployment, Re-Position Artillery, Melee Shrouding
            # upgrades are dropped.
            UpgradeSlot("aura_command", "Command aura (one)", [
                UpgradeOption("lieutenant", 15, adds_rapid_rush_aura=True),
                UpgradeOption("preacher", 20, adds_bane_melee_aura=True),
                UpgradeOption("captain", 30, adds_bane_shoot_aura=True),
            ]),
            UpgradeSlot("mobility", "Add mobility upgrade", [
                UpgradeOption("jetpack", 20, adds_flying=True),
                UpgradeOption("combat_bike", 105, adds_fast=True, adds_tough=3,
                              adds_weapons=[BB_W_TWIN_HEAVY_RIFLE]),
            ]),
            UpgradeSlot("loadout", "Replace Combat Shield and CCW", [
                UpgradeOption("flamer_pistol", 5, removes_weapon="CCW",
                              adds_weapons=[BB_W_FLAMER_PISTOL, BB_W_CCW_A2],
                              removes_shielded=True),
                UpgradeOption("heavy_chainsaw", 15, removes_weapon="CCW",
                              adds_weapons=[BB_W_HEAVY_CHAINSAW],
                              removes_shielded=True),
                UpgradeOption("dual_energy_claws", 15, removes_weapon="CCW",
                              adds_weapons=[BB_W_DUAL_ENERGY_CLAWS_A4],
                              removes_shielded=True),
            ]),
            UpgradeSlot("ranged_swap", "Add a ranged weapon", [
                UpgradeOption("master_heavy_rifle", 30,
                              adds_weapons=[BB_W_MASTER_HEAVY_RIFLE]),
                UpgradeOption("master_storm_rifle", 70,
                              adds_weapons=[BB_W_MASTER_STORM_RIFLE]),
                UpgradeOption("master_heavy_pistol", 10,
                              adds_weapons=[BB_W_MASTER_HEAVY_PISTOL]),
            ]),
            UpgradeSlot("rifle_attachment", "Master Heavy Rifle attachment (Limited)", [
                UpgradeOption("flamer_mod", 10, adds_weapons=[BB_W_FLAMER_MOD],
                              requires="master_heavy_rifle"),
                UpgradeOption("plasma_mod", 10, adds_weapons=[BB_W_PLASMA_MOD],
                              requires="master_heavy_rifle"),
                UpgradeOption("fusion_mod", 10, adds_weapons=[BB_W_FUSION_MOD],
                              requires="master_heavy_rifle"),
                UpgradeOption("gravity_mod", 15, adds_weapons=[BB_W_GRAVITY_MOD],
                              requires="master_heavy_rifle"),
            ]),
        ],
    ))

    # Master Brother [1] — 55pts
    templates.append(UnitTemplate(
        "bb_master_brother", "Master Brother", 55, 1, quality=3, defense=3,
        tough=3, fearless=True, hero=True, faction="bb", battleborn=True,
        base_weapons=[BB_W_FLAMER_PISTOL, BB_W_CCW_A2],
        upgrade_slots=[
            UpgradeSlot("aura_command", "Command aura (one)", [
                UpgradeOption("lieutenant", 15, adds_rapid_rush_aura=True),
                UpgradeOption("preacher", 20, adds_bane_melee_aura=True),
                UpgradeOption("captain", 30, adds_bane_shoot_aura=True),
            ]),
            UpgradeSlot("ranged_swap", "Replace Flamer Pistol", [
                UpgradeOption("master_heavy_pistol", 10,
                              removes_weapon="Flamer Pistol",
                              adds_weapons=[BB_W_MASTER_HEAVY_PISTOL]),
                UpgradeOption("plasma_pistol", 5,
                              removes_weapon="Flamer Pistol",
                              adds_weapons=[BB_W_PLASMA_PISTOL]),
                UpgradeOption("fusion_pistol", 5,
                              removes_weapon="Flamer Pistol",
                              adds_weapons=[BB_W_FUSION_PISTOL]),
                UpgradeOption("gravity_pistol", 5,
                              removes_weapon="Flamer Pistol",
                              adds_weapons=[BB_W_GRAVITY_PISTOL]),
                UpgradeOption("master_heavy_rifle", 20,
                              removes_weapon="Flamer Pistol",
                              adds_weapons=[BB_W_MASTER_HEAVY_RIFLE]),
                UpgradeOption("master_storm_rifle", 50,
                              removes_weapon="Flamer Pistol",
                              adds_weapons=[BB_W_MASTER_STORM_RIFLE]),
            ]),
            UpgradeSlot("rifle_attachment", "Master Heavy Rifle attachment (Limited)", [
                UpgradeOption("flamer_mod", 5, adds_weapons=[BB_W_FLAMER_MOD],
                              requires="master_heavy_rifle"),
                UpgradeOption("plasma_mod", 10, adds_weapons=[BB_W_PLASMA_MOD],
                              requires="master_heavy_rifle"),
                UpgradeOption("fusion_mod", 10, adds_weapons=[BB_W_FUSION_MOD],
                              requires="master_heavy_rifle"),
                UpgradeOption("gravity_mod", 10, adds_weapons=[BB_W_GRAVITY_MOD],
                              requires="master_heavy_rifle"),
            ]),
            UpgradeSlot("ccw_swap", "Replace CCW", [
                UpgradeOption("energy_sword", 10, removes_weapon="CCW",
                              adds_weapons=[BB_W_ENERGY_SWORD_A2]),
                UpgradeOption("chain_fist", 15, removes_weapon="CCW",
                              adds_weapons=[BB_W_CHAIN_FIST_A1]),
                UpgradeOption("energy_fist", 15, removes_weapon="CCW",
                              adds_weapons=[BB_W_ENERGY_FIST_A2]),
            ]),
            UpgradeSlot("mobility", "Add mobility", [
                UpgradeOption("jetpack", 20, adds_flying=True),
            ]),
        ],
    ))

    # Elite Pathfinder [1] — 55pts (Strider)
    templates.append(UnitTemplate(
        "bb_elite_pathfinder", "Elite Pathfinder", 55, 1, quality=4, defense=4,
        tough=3, fearless=True, hero=True, faction="bb", battleborn=True,
        strider=True,
        base_weapons=[BB_W_FLAMER_PISTOL, BB_W_CCW_A2],
        upgrade_slots=[
            UpgradeSlot("aura_command", "Command aura (one)", [
                UpgradeOption("preacher", 20, adds_bane_melee_aura=True),
                UpgradeOption("captain", 30, adds_bane_shoot_aura=True),
            ]),
            UpgradeSlot("ranged_swap", "Replace Flamer Pistol", [
                UpgradeOption("master_shotgun", 5,
                              removes_weapon="Flamer Pistol",
                              adds_weapons=[BB_W_MASTER_SHOTGUN]),
                UpgradeOption("master_sniper_rifle", 50,
                              removes_weapon="Flamer Pistol",
                              adds_weapons=[BB_W_MASTER_SNIPER_RIFLE]),
                UpgradeOption("master_heavy_rifle", 15,
                              removes_weapon="Flamer Pistol",
                              adds_weapons=[BB_W_MASTER_HEAVY_RIFLE]),
            ]),
            UpgradeSlot("rifle_attachment", "Master Heavy Rifle attachment (Limited)", [
                UpgradeOption("flamer_mod", 5, adds_weapons=[BB_W_FLAMER_MOD],
                              requires="master_heavy_rifle"),
                UpgradeOption("plasma_mod", 5, adds_weapons=[BB_W_PLASMA_MOD],
                              requires="master_heavy_rifle"),
                UpgradeOption("fusion_mod", 5, adds_weapons=[BB_W_FUSION_MOD],
                              requires="master_heavy_rifle"),
                UpgradeOption("gravity_mod", 10, adds_weapons=[BB_W_GRAVITY_MOD],
                              requires="master_heavy_rifle"),
            ]),
            UpgradeSlot("stealth", "Stealth/Scout option", [
                UpgradeOption("forward_sentry", 10, adds_scout=True),
                UpgradeOption("camo_cloak", 10, adds_stealth=True),
            ]),
            UpgradeSlot("mobility", "Combat Bike", [
                UpgradeOption("combat_bike", 75, adds_fast=True, adds_tough=3,
                              adds_weapons=[BB_W_TWIN_HEAVY_RIFLE]),
            ]),
            UpgradeSlot("ccw_swap", "Replace CCW", [
                UpgradeOption("energy_sword", 5, removes_weapon="CCW",
                              adds_weapons=[BB_W_ENERGY_SWORD_A2]),
                UpgradeOption("energy_fist", 10, removes_weapon="CCW",
                              adds_weapons=[BB_W_ENERGY_FIST_A2]),
            ]),
        ],
    ))

    # ---------- INFANTRY ----------

    # Pathfinders [5] — 120pts (Strider)
    templates.append(UnitTemplate(
        "bb_pathfinders", "Pathfinders", 120, 5, quality=4, defense=4,
        fearless=True, faction="bb", battleborn=True, strider=True,
        base_weapons=[BB_W_HEAVY_PISTOL] * 5 + [BB_W_CCW_A2] * 5,
        upgrade_slots=[
            UpgradeSlot("primary_swap_all", "Replace all Heavy Pistols and CCWs", [
                UpgradeOption("heavy_rifles", 10,
                              removes_weapon="Heavy Pistol", removes_count=5,
                              adds_weapons=[BB_W_HEAVY_RIFLE] * 5,
                              applies_to_all=True,
                              removes_weapon_2="CCW", removes_count_2=5,
                              adds_weapons_2=[BB_W_CCW_A1] * 5),
                UpgradeOption("shotguns", 10,
                              removes_weapon="Heavy Pistol", removes_count=5,
                              adds_weapons=[BB_W_SHOTGUN] * 5,
                              applies_to_all=True,
                              removes_weapon_2="CCW", removes_count_2=5,
                              adds_weapons_2=[BB_W_CCW_A1] * 5),
            ]),
            UpgradeSlot("specialist", "Replace one Heavy Pistol", [
                UpgradeOption("gravity_rifle", 5,
                              removes_weapon="Heavy Pistol", removes_count=1,
                              adds_weapons=[BB_W_GRAVITY_RIFLE]),
                UpgradeOption("plasma_rifle", 10,
                              removes_weapon="Heavy Pistol", removes_count=1,
                              adds_weapons=[BB_W_PLASMA_RIFLE]),
                UpgradeOption("flamer", 15,
                              removes_weapon="Heavy Pistol", removes_count=1,
                              adds_weapons=[BB_W_FLAMER]),
                UpgradeOption("heavy_machinegun", 30,
                              removes_weapon="Heavy Pistol", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_MACHINEGUN]),
                UpgradeOption("missile_launcher", 35,
                              removes_weapon="Heavy Pistol", removes_count=1,
                              adds_weapons=[BB_W_MISSILE_LAUNCHER]),
            ]),
            UpgradeSlot("snipers", "Replace up to three Heavy Rifles with Sniper Rifles", [
                UpgradeOption("sniper_rifles_1", 20,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_SNIPER_RIFLE],
                              requires="heavy_rifles"),
            ]),
            UpgradeSlot("stealth", "Upgrade all models", [
                UpgradeOption("forward_sentries", 15, adds_scout=True,
                              applies_to_all=True),
                UpgradeOption("camo_cloaks", 15, adds_stealth=True,
                              applies_to_all=True),
            ]),
            UpgradeSlot("sgt_loadout", "Sergeant loadout", [
                UpgradeOption("sgt_pistol_sword", 5,
                              removes_weapon="Heavy Pistol", removes_count=1,
                              adds_weapons=[BB_W_SGT_HEAVY_PISTOL,
                                            BB_W_ENERGY_SWORD_A2],
                              removes_weapon_2="CCW", removes_count_2=1),
            ]),
        ],
    ))

    # Battle Brothers [5] — 150pts
    templates.append(UnitTemplate(
        "bb_battle_brothers", "Battle Brothers", 150, 5, quality=3, defense=3,
        fearless=True, faction="bb", battleborn=True,
        base_weapons=[BB_W_HEAVY_RIFLE] * 5 + [BB_W_CCW_A1] * 5,
        upgrade_slots=[
            UpgradeSlot("banner", "Upgrade one model with banner", [
                UpgradeOption("founders_banner", 10,
                              adds_versatile_reach_aura=True),
                UpgradeOption("detachment_banner", 10,
                              adds_courage_aura=True),
                UpgradeOption("medical_training", 20,
                              adds_regeneration_aura=True),
            ]),
            UpgradeSlot("heavy_swap", "Replace one Heavy Rifle", [
                UpgradeOption("flamer", 5,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_FLAMER]),
                UpgradeOption("gravity_rifle", 5,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_GRAVITY_RIFLE]),
                UpgradeOption("plasma_rifle", 10,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_PLASMA_RIFLE]),
                UpgradeOption("heavy_flamer", 10,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_FLAMER]),
                UpgradeOption("fusion_rifle", 15,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_FUSION_RIFLE]),
                UpgradeOption("gravity_cannon", 15,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_GRAVITY_CANNON]),
                UpgradeOption("heavy_fusion_rifle", 30,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_FUSION_RIFLE]),
                UpgradeOption("heavy_machinegun", 35,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_MACHINEGUN]),
                UpgradeOption("missile_launcher", 40,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_MISSILE_LAUNCHER]),
                UpgradeOption("plasma_cannon", 50,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_PLASMA_CANNON]),
                UpgradeOption("laser_cannon", 55,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_LASER_CANNON]),
            ]),
            UpgradeSlot("sgt_loadout", "Sergeant loadout", [
                UpgradeOption("sgt_pistol_sword", 5,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_SGT_HEAVY_PISTOL,
                                            BB_W_ENERGY_SWORD_A2],
                              removes_weapon_2="CCW", removes_count_2=1),
                UpgradeOption("sgt_plasma_pistol", 10,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_PLASMA_PISTOL,
                                            BB_W_ENERGY_SWORD_A2],
                              removes_weapon_2="CCW", removes_count_2=1),
                UpgradeOption("sgt_energy_fist", 5,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_SGT_HEAVY_PISTOL,
                                            BB_W_ENERGY_FIST_A2],
                              removes_weapon_2="CCW", removes_count_2=1),
            ]),
        ],
    ))

    # Assault Brothers [5] — 165pts
    templates.append(UnitTemplate(
        "bb_assault_brothers", "Assault Brothers", 165, 5, quality=3, defense=3,
        fearless=True, faction="bb", battleborn=True,
        base_weapons=[BB_W_HEAVY_PISTOL] * 5 + [BB_W_HEAVY_CCW] * 5,
        upgrade_slots=[
            UpgradeSlot("banner", "Upgrade one model with banner", [
                UpgradeOption("founders_banner", 15,
                              adds_versatile_reach_aura=True),
                UpgradeOption("detachment_banner", 10,
                              adds_courage_aura=True),
                UpgradeOption("medical_training", 20,
                              adds_regeneration_aura=True),
            ]),
            UpgradeSlot("ccw_swap", "Replace one Heavy Pistol and Heavy CCW", [
                UpgradeOption("hammer_combo", 0,
                              removes_weapon="Heavy Pistol", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_PISTOL,
                                            BB_W_ENERGY_HAMMER_A1],
                              removes_weapon_2="Heavy CCW", removes_count_2=1),
                UpgradeOption("sword_combo", 5,
                              removes_weapon="Heavy Pistol", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_PISTOL,
                                            BB_W_ENERGY_SWORD_A2],
                              removes_weapon_2="Heavy CCW", removes_count_2=1),
                UpgradeOption("dual_claws", 5,
                              removes_weapon="Heavy CCW", removes_count=1,
                              adds_weapons=[BB_W_DUAL_ENERGY_CLAWS_A4]),
                UpgradeOption("chainsaw", 5,
                              removes_weapon="Heavy CCW", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_CHAINSAW]),
                UpgradeOption("fist_combo", 10,
                              removes_weapon="Heavy Pistol", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_PISTOL,
                                            BB_W_ENERGY_FIST_A2],
                              removes_weapon_2="Heavy CCW", removes_count_2=1),
            ]),
            UpgradeSlot("jetpacks", "Add Jetpacks to all models", [
                UpgradeOption("jetpacks", 35, adds_flying=True,
                              applies_to_all=True),
            ]),
            UpgradeSlot("sgt_loadout", "Sergeant loadout", [
                UpgradeOption("sgt_pistol_sword", 5,
                              removes_weapon="Heavy Pistol", removes_count=1,
                              adds_weapons=[BB_W_SGT_HEAVY_PISTOL,
                                            BB_W_ENERGY_SWORD_A2],
                              removes_weapon_2="Heavy CCW", removes_count_2=1),
            ]),
        ],
    ))

    # Veteran Assault Brothers [3] — 90pts (Shielded, Versatile Attack)
    templates.append(UnitTemplate(
        "bb_vet_assault_brothers", "Veteran Assault Brothers", 90, 3, quality=3, defense=3,
        fearless=True, shielded=True, faction="bb", battleborn=True,
        versatile_attack=True,
        base_weapons=[BB_W_HEAVY_PISTOL] * 3 + [BB_W_BASH_A1] * 3,
        upgrade_slots=[
            UpgradeSlot("banner", "Upgrade one model with banner", [
                UpgradeOption("founders_banner", 5,
                              adds_versatile_reach_aura=True),
                UpgradeOption("detachment_banner", 5,
                              adds_courage_aura=True),
                UpgradeOption("medical_training", 15,
                              adds_regeneration_aura=True),
            ]),
            UpgradeSlot("ccw_swap", "Replace any Heavy Pistol and Combat Shield", [
                UpgradeOption("heavy_ccw_shield", 5,
                              removes_weapon="Bash", removes_count=3,
                              adds_weapons=[BB_W_HEAVY_CCW] * 3,
                              applies_to_all=True),
                UpgradeOption("dual_claws_all", 20,
                              removes_weapon="Bash", removes_count=3,
                              adds_weapons=[BB_W_DUAL_ENERGY_CLAWS_A4] * 3,
                              applies_to_all=True,
                              removes_shielded=True),
                UpgradeOption("chainsaw_all", 20,
                              removes_weapon="Bash", removes_count=3,
                              adds_weapons=[BB_W_HEAVY_CHAINSAW] * 3,
                              applies_to_all=True,
                              removes_shielded=True),
            ]),
            UpgradeSlot("jetpacks", "Add Jetpacks to all models", [
                UpgradeOption("jetpacks", 20, adds_flying=True,
                              applies_to_all=True),
            ]),
            UpgradeSlot("sgt_loadout", "Sergeant loadout", [
                UpgradeOption("sgt_pistol_sword", 15,
                              removes_weapon="Heavy Pistol", removes_count=1,
                              adds_weapons=[BB_W_SGT_HEAVY_PISTOL,
                                            BB_W_ENERGY_SWORD_A2],
                              removes_weapon_2="Bash", removes_count_2=1),
            ]),
        ],
    ))

    # Veteran Battle Brothers [3] — 115pts (Versatile Attack)
    templates.append(UnitTemplate(
        "bb_vet_battle_brothers", "Veteran Battle Brothers", 115, 3, quality=3, defense=3,
        fearless=True, faction="bb", battleborn=True, versatile_attack=True,
        base_weapons=[BB_W_HEAVY_RIFLE] * 3 + [BB_W_CCW_A1] * 3,
        upgrade_slots=[
            UpgradeSlot("banner", "Upgrade one model with banner", [
                UpgradeOption("founders_banner", 5,
                              adds_versatile_reach_aura=True),
                UpgradeOption("detachment_banner", 5,
                              adds_courage_aura=True),
                UpgradeOption("medical_training", 15,
                              adds_regeneration_aura=True),
            ]),
            UpgradeSlot("heavy_swap", "Replace one Heavy Rifle", [
                UpgradeOption("flamer", 5,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_FLAMER]),
                UpgradeOption("gravity_rifle", 5,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_GRAVITY_RIFLE]),
                UpgradeOption("heavy_flamer", 10,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_FLAMER]),
                UpgradeOption("fusion_rifle", 15,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_FUSION_RIFLE]),
                UpgradeOption("plasma_rifle", 10,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_PLASMA_RIFLE]),
                UpgradeOption("heavy_fusion_rifle", 35,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_FUSION_RIFLE]),
                UpgradeOption("missile_launcher", 50,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_MISSILE_LAUNCHER]),
                UpgradeOption("plasma_cannon", 55,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_PLASMA_CANNON]),
                UpgradeOption("laser_cannon", 65,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_LASER_CANNON]),
            ]),
            UpgradeSlot("rifle_attachment", "Master Heavy Rifle attachment (Limited)", [
                UpgradeOption("flamer_mod", 10, adds_weapons=[BB_W_FLAMER_MOD]),
                UpgradeOption("plasma_mod", 10, adds_weapons=[BB_W_PLASMA_MOD]),
                UpgradeOption("fusion_mod", 10, adds_weapons=[BB_W_FUSION_MOD]),
                UpgradeOption("gravity_mod", 15, adds_weapons=[BB_W_GRAVITY_MOD]),
            ]),
            UpgradeSlot("sgt_loadout", "Sergeant loadout", [
                UpgradeOption("sgt_pistol_sword", 5,
                              removes_weapon="Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_SGT_HEAVY_PISTOL,
                                            BB_W_ENERGY_SWORD_A2],
                              removes_weapon_2="CCW", removes_count_2=1),
            ]),
        ],
    ))

    # Support Brothers [3] — 130pts (Relentless)
    templates.append(UnitTemplate(
        "bb_support_brothers", "Support Brothers", 130, 3, quality=3, defense=3,
        fearless=True, relentless=True, faction="bb", battleborn=True,
        base_weapons=[BB_W_HEAVY_FLAMER] * 3 + [BB_W_CCW_A1] * 3,
        upgrade_slots=[
            UpgradeSlot("heavy_swap_all", "Replace any Heavy Flamer", [
                UpgradeOption("gravity_cannon", 5,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_GRAVITY_CANNON]),
                UpgradeOption("heavy_fusion_rifle", 20,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_FUSION_RIFLE]),
                UpgradeOption("heavy_machinegun", 25,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_MACHINEGUN]),
                UpgradeOption("missile_launcher", 30,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_MISSILE_LAUNCHER]),
                UpgradeOption("plasma_cannon", 40,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_PLASMA_CANNON]),
                UpgradeOption("laser_cannon", 45,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_LASER_CANNON]),
            ]),
        ],
    ))

    # Destroyers [3] — 220pts (Ambush dropped, Shielded, Tough(3))
    templates.append(UnitTemplate(
        "bb_destroyers", "Destroyers", 220, 3, quality=3, defense=3,
        tough=3, fearless=True, shielded=True, faction="bb", battleborn=True,
        base_weapons=[BB_W_CCW_A3] * 3,
        upgrade_slots=[
            UpgradeSlot("banner", "Upgrade one model with banner", [
                UpgradeOption("founders_banner", 5,
                              adds_versatile_reach_aura=True),
                UpgradeOption("detachment_banner", 25,
                              adds_courage_aura=True),
                UpgradeOption("medical_training", 55,
                              adds_regeneration_aura=True),
            ]),
            UpgradeSlot("loadout_swap_all", "Replace all Combat Shields and CCWs", [
                UpgradeOption("dual_claws_all", 15,
                              removes_weapon="CCW", removes_count=3,
                              adds_weapons=[BB_W_DUAL_ENERGY_CLAWS_A4] * 3,
                              applies_to_all=True,
                              removes_shielded=True),
                UpgradeOption("storm_rifle_ccw_all", 80,
                              removes_weapon="CCW", removes_count=3,
                              adds_weapons=[BB_W_STORM_RIFLE_A3,
                                            BB_W_STORM_RIFLE_A3,
                                            BB_W_STORM_RIFLE_A3,
                                            BB_W_CCW_A1, BB_W_CCW_A1, BB_W_CCW_A1],
                              applies_to_all=True,
                              removes_shielded=True),
            ]),
            UpgradeSlot("ccw_replace", "Replace any CCW", [
                UpgradeOption("energy_hammer", 0,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[BB_W_ENERGY_HAMMER_A3]),
                UpgradeOption("chain_fist", 10,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[BB_W_CHAIN_FIST_A1]),
                UpgradeOption("energy_sword", 10,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[BB_W_ENERGY_SWORD_A3]),
                UpgradeOption("energy_fist", 20,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[BB_W_ENERGY_FIST_A3]),
            ]),
        ],
    ))

    # ---------- BIKES ----------

    # Pathfinder Bikers [3] — 205pts (Fast, Tough(3))
    templates.append(UnitTemplate(
        "bb_pathfinder_bikers", "Pathfinder Bikers", 205, 3, quality=4, defense=4,
        tough=3, fast=True, fearless=True, faction="bb", battleborn=True,
        base_weapons=[BB_W_GRENADE_LAUNCHER] * 3 + [BB_W_HEAVY_PISTOL] * 3
                     + [BB_W_CCW_A2] * 3,
        upgrade_slots=[
            UpgradeSlot("grenade_swap", "Replace any Grenade Launcher", [
                UpgradeOption("twin_heavy_rifle_all", 5,
                              removes_weapon="Grenade Launcher", removes_count=3,
                              adds_weapons=[BB_W_TWIN_HEAVY_RIFLE] * 3,
                              applies_to_all=True),
            ]),
            UpgradeSlot("primary_swap_all", "Replace all Heavy Pistols and CCWs", [
                UpgradeOption("heavy_rifles_all", 5,
                              removes_weapon="Heavy Pistol", removes_count=3,
                              adds_weapons=[BB_W_HEAVY_RIFLE] * 3,
                              applies_to_all=True,
                              removes_weapon_2="CCW", removes_count_2=3,
                              adds_weapons_2=[BB_W_CCW_A1] * 3),
            ]),
            UpgradeSlot("tires", "Off-Road Tires (Strider)", [
                UpgradeOption("offroad_tires", 25, adds_strider=True),
            ]),
        ],
    ))

    # Brother Bikers [3] — 280pts (Fast, Tough(3))
    templates.append(UnitTemplate(
        "bb_brother_bikers", "Brother Bikers", 280, 3, quality=3, defense=3,
        tough=3, fast=True, fearless=True, faction="bb", battleborn=True,
        base_weapons=[BB_W_TWIN_HEAVY_RIFLE] * 3 + [BB_W_HEAVY_PISTOL] * 3
                     + [BB_W_CCW_A2] * 3,
        upgrade_slots=[
            UpgradeSlot("primary_swap_all", "Replace all Heavy Pistols and CCWs", [
                UpgradeOption("heavy_rifles_all", 5,
                              removes_weapon="Heavy Pistol", removes_count=3,
                              adds_weapons=[BB_W_HEAVY_RIFLE] * 3,
                              applies_to_all=True,
                              removes_weapon_2="CCW", removes_count_2=3,
                              adds_weapons_2=[BB_W_CCW_A1] * 3),
            ]),
            UpgradeSlot("rifle_swap", "Replace one Twin Heavy Rifle", [
                UpgradeOption("flamer", 5,
                              removes_weapon="Twin Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_FLAMER]),
                UpgradeOption("gravity_rifle", 5,
                              removes_weapon="Twin Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_GRAVITY_RIFLE]),
                UpgradeOption("plasma_rifle", 10,
                              removes_weapon="Twin Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_PLASMA_RIFLE]),
                UpgradeOption("fusion_rifle", 15,
                              removes_weapon="Twin Heavy Rifle", removes_count=1,
                              adds_weapons=[BB_W_FUSION_RIFLE]),
            ]),
        ],
    ))

    # Support Bike [1] — 175pts (Fast, Tough(6))
    templates.append(UnitTemplate(
        "bb_support_bike", "Support Bike", 175, 1, quality=3, defense=3,
        tough=6, fast=True, fearless=True, faction="bb", battleborn=True,
        base_weapons=[BB_W_TWIN_HEAVY_RIFLE, BB_W_HEAVY_FLAMER,
                      BB_W_HEAVY_PISTOL, BB_W_CCW_A3],
        upgrade_slots=[
            UpgradeSlot("targeting", "Targeting Array", [
                UpgradeOption("unstoppable_mark", 25, adds_unstoppable_mark=True),
            ]),
            UpgradeSlot("flamer_swap", "Replace Heavy Flamer", [
                UpgradeOption("heavy_fusion_rifle", 20,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_FUSION_RIFLE]),
                UpgradeOption("heavy_machinegun", 25,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_MACHINEGUN]),
            ]),
        ],
    ))

    # ---------- VEHICLES ----------

    # APC [1] — 200pts (Fast, Impact(3), Tough(6); Transport ignored)
    templates.append(UnitTemplate(
        "bb_apc", "APC", 200, 1, quality=3, defense=2,
        tough=6, fast=True, fearless=True, impact=3, faction="bb", battleborn=True,
        base_weapons=[BB_W_STORM_RIFLE_A3],
        upgrade_slots=[
            UpgradeSlot("turret", "Upgrade with one", [
                UpgradeOption("storm_rifle", 45,
                              adds_weapons=[BB_W_STORM_RIFLE_A3]),
                UpgradeOption("heavy_fusion_rifle", 45,
                              adds_weapons=[BB_W_HEAVY_FUSION_RIFLE]),
            ]),
            UpgradeSlot("hunter_missiles", "Hunter Missiles (Limited)", [
                UpgradeOption("hunter_missiles", 15,
                              adds_weapons=[BB_W_HUNTER_MISSILES]),
            ]),
            UpgradeSlot("dozer_blade", "Dozer Blade (Strider)", [
                UpgradeOption("dozer_blade", 15, adds_strider=True),
            ]),
        ],
    ))

    # Attack APC [1] — 195pts
    templates.append(UnitTemplate(
        "bb_attack_apc", "Attack APC", 195, 1, quality=3, defense=2,
        tough=6, fast=True, fearless=True, impact=3, faction="bb", battleborn=True,
        base_weapons=[BB_W_TWIN_HEAVY_FLAMER],
        upgrade_slots=[
            UpgradeSlot("turret_swap", "Replace Twin Heavy Flamer", [
                UpgradeOption("twin_heavy_machinegun", 45,
                              removes_weapon="Twin Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_TWIN_HEAVY_MACHINEGUN]),
                UpgradeOption("laser_plasma", 60,
                              removes_weapon="Twin Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_LASER_CANNON,
                                            Weapon("Twin Plasma Rifle", 24, 2, ap=4)]),
                UpgradeOption("twin_minigun", 65,
                              removes_weapon="Twin Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_TWIN_MINIGUN]),
                UpgradeOption("twin_laser_cannon", 80,
                              removes_weapon="Twin Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_TWIN_LASER_CANNON]),
            ]),
            UpgradeSlot("dozer_blade", "Dozer Blade (Strider)", [
                UpgradeOption("dozer_blade", 15, adds_strider=True),
            ]),
        ],
    ))

    # Attack Speeder [1] — 220pts (Ambush dropped, Fast, Impact(3), Strider, Tough(6))
    templates.append(UnitTemplate(
        "bb_attack_speeder", "Attack Speeder", 220, 1, quality=3, defense=2,
        tough=6, fast=True, fearless=True, impact=3, strider=True,
        faction="bb", battleborn=True,
        base_weapons=[BB_W_HEAVY_FLAMER, BB_W_HEAVY_FLAMER],
        upgrade_slots=[
            UpgradeSlot("flamer_swap_all", "Replace any Heavy Flamer", [
                UpgradeOption("heavy_fusion_rifle_all", 20,
                              removes_weapon="Heavy Flamer", removes_count=2,
                              adds_weapons=[BB_W_HEAVY_FUSION_RIFLE,
                                            BB_W_HEAVY_FUSION_RIFLE],
                              applies_to_all=True),
                UpgradeOption("heavy_machinegun_all", 25,
                              removes_weapon="Heavy Flamer", removes_count=2,
                              adds_weapons=[BB_W_HEAVY_MACHINEGUN,
                                            BB_W_HEAVY_MACHINEGUN],
                              applies_to_all=True),
                UpgradeOption("minigun_all", 35,
                              removes_weapon="Heavy Flamer", removes_count=2,
                              adds_weapons=[BB_W_MINIGUN, BB_W_MINIGUN],
                              applies_to_all=True),
            ]),
            UpgradeSlot("typhoon", "Replace one Heavy Flamer", [
                UpgradeOption("twin_typhoon_missiles", 55,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[BB_W_TWIN_TYPHOON_MISSILES]),
            ]),
            UpgradeSlot("targeting", "Targeting Array", [
                UpgradeOption("unstoppable_mark", 25, adds_unstoppable_mark=True),
            ]),
        ],
    ))

    # Heavy Exo-Suit [1] — 160pts (Fear(1), Tough(6))
    templates.append(UnitTemplate(
        "bb_heavy_exosuit", "Heavy Exo-Suit", 160, 1, quality=3, defense=2,
        tough=6, fearless=True, fear=1, faction="bb", battleborn=True,
        base_weapons=[BB_W_TWIN_FLAMER, BB_W_STOMP_A2],
        upgrade_slots=[
            UpgradeSlot("ranged_swap", "Replace Twin Flamer", [
                UpgradeOption("twin_fusion_rifle", 20,
                              removes_weapon="Twin Flamer", removes_count=1,
                              adds_weapons=[BB_W_TWIN_FUSION_RIFLE]),
                UpgradeOption("twin_light_gravity_cannon", 20,
                              removes_weapon="Twin Flamer", removes_count=1,
                              adds_weapons=[BB_W_TWIN_LIGHT_GRAVITY_CANNON]),
                UpgradeOption("twin_heavy_machinegun", 65,
                              removes_weapon="Twin Flamer", removes_count=1,
                              adds_weapons=[BB_W_TWIN_HEAVY_MACHINEGUN]),
                UpgradeOption("twin_laser_cannon", 100,
                              removes_weapon="Twin Flamer", removes_count=1,
                              adds_weapons=[BB_W_TWIN_LASER_CANNON]),
            ]),
            UpgradeSlot("melee_extra", "Melee upgrade", [
                UpgradeOption("dual_heavy_fists", 20,
                              adds_weapons=[BB_W_DUAL_HEAVY_FISTS]),
                UpgradeOption("dual_combat_drills", 45,
                              adds_weapons=[BB_W_DUAL_COMBAT_DRILLS]),
            ]),
            UpgradeSlot("chest", "Chest weapon", [
                UpgradeOption("chest_missiles", 20,
                              adds_weapons=[BB_W_CHEST_MISSILES]),
                UpgradeOption("chest_rifles", 30,
                              adds_weapons=[BB_W_CHEST_RIFLES]),
            ]),
        ],
    ))

    # Battle Tank [1] — 465pts (Fast, Impact(6), Tough(12); Transport ignored)
    templates.append(UnitTemplate(
        "bb_battle_tank", "Battle Tank", 465, 1, quality=3, defense=2,
        tough=12, fast=True, fearless=True, impact=6, faction="bb", battleborn=True,
        base_weapons=[BB_W_TWIN_HEAVY_MACHINEGUN, BB_W_TWIN_STORM_CANNON],
        upgrade_slots=[
            UpgradeSlot("primary_swap", "Replace Twin Storm Cannon", [
                UpgradeOption("spear_missile_launcher", 5,
                              removes_weapon="Twin Storm Cannon", removes_count=1,
                              adds_weapons=[BB_W_SPEAR_MISSILE_LAUNCHER]),
                UpgradeOption("demolition_cannon", 10,
                              removes_weapon="Twin Storm Cannon", removes_count=1,
                              adds_weapons=[BB_W_DEMOLITION_CANNON]),
                UpgradeOption("wind_missile_launcher", 35,
                              removes_weapon="Twin Storm Cannon", removes_count=1,
                              adds_weapons=[BB_W_WIND_MISSILE_LAUNCHER]),
                UpgradeOption("twin_laser_cannon", 45,
                              removes_weapon="Twin Storm Cannon", removes_count=1,
                              adds_weapons=[BB_W_TWIN_LASER_CANNON]),
                UpgradeOption("rapid_autocannon", 50,
                              removes_weapon="Twin Storm Cannon", removes_count=1,
                              adds_weapons=[BB_W_RAPID_AUTOCANNON]),
            ]),
            UpgradeSlot("secondary_swap", "Replace Twin Heavy Machineguns", [
                UpgradeOption("twin_laser_cannon_sec", 35,
                              removes_weapon="Twin Heavy Machinegun", removes_count=1,
                              adds_weapons=[BB_W_TWIN_LASER_CANNON]),
            ]),
            UpgradeSlot("turret", "Upgrade with one", [
                UpgradeOption("storm_rifle", 45,
                              adds_weapons=[BB_W_STORM_RIFLE_A3]),
                UpgradeOption("heavy_fusion_rifle", 45,
                              adds_weapons=[BB_W_HEAVY_FUSION_RIFLE]),
            ]),
            UpgradeSlot("hunter_missiles", "Hunter Missiles (Limited)", [
                UpgradeOption("hunter_missiles", 15,
                              adds_weapons=[BB_W_HUNTER_MISSILES]),
            ]),
            UpgradeSlot("dozer_blade", "Dozer Blade (Strider)", [
                UpgradeOption("dozer_blade", 30, adds_strider=True),
            ]),
        ],
    ))

    # Combat Walker [1] — 350pts (Fear(2), Tough(12))
    templates.append(UnitTemplate(
        "bb_combat_walker", "Combat Walker", 350, 1, quality=3, defense=2,
        tough=12, fearless=True, fear=2, faction="bb", battleborn=True,
        base_weapons=[BB_W_STOMP_A4, BB_W_WALKER_FIST, BB_W_WALKER_FIST],
        upgrade_slots=[
            UpgradeSlot("fist_left", "Replace one Walker Fist (a)", [
                UpgradeOption("twin_heavy_flamer_a", 5,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_TWIN_HEAVY_FLAMER]),
                UpgradeOption("heavy_plasma_cannon_a", 45,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_PLASMA_CANNON]),
                UpgradeOption("heavy_rifle_array_a", 40,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_RIFLE_ARRAY]),
                UpgradeOption("twin_heavy_machinegun_a", 55,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_TWIN_HEAVY_MACHINEGUN]),
                UpgradeOption("twin_laser_cannon_a", 90,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_TWIN_LASER_CANNON]),
            ]),
            UpgradeSlot("fist_right", "Replace one Walker Fist (b)", [
                UpgradeOption("missile_array_b", 55,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_MISSILE_ARRAY]),
                UpgradeOption("twin_autocannon_b", 95,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_TWIN_AUTOCANNON]),
            ]),
        ],
    ))

    # Veteran Combat Walker [1] — 385pts (Fear(2), Tough(12), Versatile Attack)
    templates.append(UnitTemplate(
        "bb_vet_combat_walker", "Veteran Combat Walker", 385, 1, quality=3, defense=2,
        tough=12, fearless=True, fear=2, faction="bb", battleborn=True,
        versatile_attack=True,
        base_weapons=[BB_W_STOMP_A4, BB_W_WALKER_FIST, BB_W_WALKER_FIST],
        upgrade_slots=[
            UpgradeSlot("fist_left", "Replace one Walker Fist (a)", [
                UpgradeOption("twin_heavy_flamer_a", 5,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_TWIN_HEAVY_FLAMER]),
                UpgradeOption("super_heavy_fusion_rifle_a", 25,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_SUPER_HEAVY_FUSION_RIFLE]),
                UpgradeOption("heavy_plasma_cannon_a", 50,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_PLASMA_CANNON]),
                UpgradeOption("twin_heavy_machinegun_a", 75,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_TWIN_HEAVY_MACHINEGUN]),
                UpgradeOption("twin_laser_cannon_a", 115,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_TWIN_LASER_CANNON]),
            ]),
            UpgradeSlot("fist_right", "Replace one Walker Fist (b)", [
                UpgradeOption("missile_array_b", 55,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_MISSILE_ARRAY]),
                UpgradeOption("twin_autocannon_b", 125,
                              removes_weapon="Walker Fist", removes_count=1,
                              adds_weapons=[BB_W_TWIN_AUTOCANNON]),
            ]),
        ],
    ))

    # Artillery Gun [1] — 195pts (Artillery, Tough(3))
    templates.append(UnitTemplate(
        "bb_artillery_gun", "Artillery Gun", 195, 1, quality=3, defense=3,
        tough=3, fearless=True, artillery=True, faction="bb", battleborn=True,
        base_weapons=[BB_W_HEAVY_GATLING_CANNON, W_ARTILLERY_CREW],
        upgrade_slots=[
            UpgradeSlot("primary_swap", "Replace Heavy Gatling Cannon", [
                UpgradeOption("heavy_flak_cannon", 50,
                              removes_weapon="Heavy Gatling Cannon", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_FLAK_CANNON]),
                UpgradeOption("heavy_thunder_cannon", 60,
                              removes_weapon="Heavy Gatling Cannon", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_THUNDER_CANNON]),
                UpgradeOption("heavy_crack_cannon", 90,
                              removes_weapon="Heavy Gatling Cannon", removes_count=1,
                              adds_weapons=[BB_W_HEAVY_CRACK_CANNON]),
            ]),
        ],
    ))

    # Heavy Tank [1] — 715pts (Fast, Impact(9), Tough(18); Transport ignored)
    templates.append(UnitTemplate(
        "bb_heavy_tank", "Heavy Tank", 715, 1, quality=3, defense=2,
        tough=18, fast=True, fearless=True, impact=9, faction="bb", battleborn=True,
        base_weapons=[BB_W_TWIN_HEAVY_MACHINEGUN, BB_W_QUAD_FLAMER_CANNON],
        upgrade_slots=[
            UpgradeSlot("turret", "Upgrade with one", [
                UpgradeOption("storm_rifle", 45,
                              adds_weapons=[BB_W_STORM_RIFLE_A3]),
                UpgradeOption("heavy_fusion_rifle", 45,
                              adds_weapons=[BB_W_HEAVY_FUSION_RIFLE]),
            ]),
            UpgradeSlot("hunter_missiles", "Hunter Missiles (Limited)", [
                UpgradeOption("hunter_missiles", 15,
                              adds_weapons=[BB_W_HUNTER_MISSILES]),
            ]),
            UpgradeSlot("dozer_blade", "Dozer Blade (Strider)", [
                UpgradeOption("dozer_blade", 45, adds_strider=True),
            ]),
            UpgradeSlot("primary_swap", "Replace Quad Flamer Cannon", [
                UpgradeOption("twin_heavy_rifle_array", 20,
                              removes_weapon="Quad Flamer Cannon", removes_count=1,
                              adds_weapons=[BB_W_TWIN_HEAVY_RIFLE_ARRAY]),
                UpgradeOption("quad_laser_cannon", 110,
                              removes_weapon="Quad Flamer Cannon", removes_count=1,
                              adds_weapons=[BB_W_QUAD_LASER_CANNON]),
            ]),
            UpgradeSlot("secondary_swap", "Replace Twin Heavy Machineguns", [
                UpgradeOption("twin_minigun_sec", 20,
                              removes_weapon="Twin Heavy Machinegun", removes_count=1,
                              adds_weapons=[BB_W_TWIN_MINIGUN]),
            ]),
        ],
    ))

    # Drop Pod, Light Gunship and Heavy Gunship are NOT included.
    return templates


# ===================================================================
# ETERNAL DYNASTY — WEAPON CONSTANTS
# ===================================================================

# Pistols / rifles
ED_W_LONG_RIFLE = Weapon("Long Rifle", 30, 1)
ED_W_DYNASTY_PISTOL = Weapon("Dynasty Pistol", 12, 2, ap=1)
ED_W_LEADER_AUTOGUN = Weapon("Leader Auto-Gun", 12, 2, rending=True)
ED_W_LEADER_CARBINE = Weapon("Leader Carbine", 18, 3)
ED_W_LEADER_LONG_RIFLE = Weapon("Leader Long Rifle", 30, 2)
ED_W_LEADER_SHOTGUN = Weapon("Leader Shotgun", 12, 3, ap=1)
ED_W_AUTOGUN = Weapon("Auto-Gun", 12, 1, rending=True)
ED_W_AUTOGUN_TWIN = Weapon("Auto-Gun (twin)", 12, 1, rending=True)  # Ninja upgrade variant
ED_W_HEAVY_PISTOL = BB_W_HEAVY_PISTOL  # 12,A1,AP(1) shared
ED_W_TWIN_HEAVY_PISTOL = Weapon("Twin Heavy Pistol", 12, 2, ap=1)
ED_W_FLAMER = Weapon("Flamer", 12, 1, blast=3, reliable=True)
ED_W_SHRED_RIFLE = Weapon("Shred Rifle", 18, 2, rending=True)
ED_W_PLASMA_RIFLE = Weapon("Plasma Rifle", 24, 1, ap=4)
ED_W_FUSION_RIFLE = Weapon("Fusion Rifle", 12, 1, ap=4, deadly=3)
ED_W_SHISHI_TURRET_ROCKET = Weapon("Shishi Turret (Rocket)", 24, 1, blast=3, indirect=True)
ED_W_SHISHI_TURRET_MISSILE = Weapon("Shishi Turret (Missiles)", 30, 2, ap=3, unstoppable=True)
ED_W_SHOTGUN = Weapon("Shotgun", 12, 2, ap=1)
ED_W_CARBINE = Weapon("Carbine", 18, 2)
ED_W_ENERGY_RIFLE = Weapon("Energy Rifle", 36, 1, ap=4)
ED_W_SNIPER_RIFLE = Weapon("Sniper Rifle", 30, 1, ap=1, reliable=True, takedown=True)
ED_W_LASER_GUN = Weapon("Laser Gun", 18, 1, tear=True)
ED_W_HEAVY_LASER_GUN = Weapon("Heavy Laser Gun", 24, 1, deadly=3, puncture=True)
ED_W_HULL_FLAMER = Weapon("Hull Flamer", 12, 2, ap=1, reliable=True)
ED_W_HULL_LASER_GUN = Weapon("Hull Laser Gun", 24, 2, tear=True)
ED_W_HULL_AUTOGUN = Weapon("Hull Auto-Gun", 18, 2, rending=True)
ED_W_TWIN_BURST_LASER_GUN = Weapon("Twin Burst Laser Gun", 18, 4, tear=True)
ED_W_TWIN_HEAVY_LASER_GUN = Weapon("Twin Heavy Laser Gun", 24, 2, deadly=3, tear=True)
ED_W_HEAVY_AUTO_GUN = Weapon("Heavy Auto-Gun", 18, 4, rending=True)
ED_W_HEAVY_LASER_GUN_TANK = Weapon("Heavy Laser Gun (tank)", 24, 1, deadly=3, puncture=True)

# Drones (per the schedule each is 1 weapon attached as an upgrade)
ED_W_DRONE_LASER_GUN = Weapon("Drone Laser Gun", 18, 1, limited=True, tear=True)

# Heavy fists / weapons for ONIs and ONI Captain
ED_W_GREAT_MACE = Weapon("Great Mace", 0, 2, blast=3, melee=True)
ED_W_HEAVY_BASH = Weapon("Heavy Bash", 0, 2, ap=1, melee=True)
ED_W_BASH_A1 = BB_W_BASH_A1  # shared
ED_W_HEAVY_FIST_A1 = Weapon("Heavy Fist", 0, 1, ap=4, melee=True)
ED_W_HEAVY_FIST_A3 = Weapon("Heavy Fist", 0, 3, ap=4, melee=True)
ED_W_HEAVY_FIST_A6 = Weapon("Heavy Fist", 0, 6, ap=1, melee=True)  # Cyber Beast
ED_W_HEAVY_GLAIVE = Weapon("Heavy Glaive", 0, 6, ap=2, melee=True)
ED_W_HEAVY_GLAIVE_A3 = Weapon("Heavy Glaive", 0, 3, ap=2, melee=True)
ED_W_HEAVY_SWORD = Weapon("Heavy Sword", 0, 4, ap=1, rending=True, melee=True)
ED_W_HEAVY_SWORD_A6 = Weapon("Heavy Sword", 0, 6, ap=1, rending=True, melee=True)
ED_W_GREAT_SWORD = Weapon("Great Sword", 0, 6, ap=1, rending=True, melee=True)
ED_W_HEAVY_FLAME_FIST = Weapon("Heavy Flame-Fist", 12, 1, ap=1, blast=3, reliable=True)
ED_W_HEAVY_GUN_FIST = Weapon("Heavy Gun-Fist", 24, 3)
ED_W_HEAVY_SHRED_FIST = Weapon("Heavy Shred-Fist", 18, 3, rending=True)
ED_W_HEAVY_FUSION_FIST = Weapon("Heavy Fusion-Fist", 18, 1, ap=4, deadly=3)
ED_W_HEAVY_ROCKET_FIST = Weapon("Heavy Rocket-Fist", 24, 1, ap=1, blast=3, indirect=True)
ED_W_HEAVY_PLASMA_FIST = Weapon("Heavy Plasma-Fist", 24, 3, ap=4)
ED_W_HEAVY_MISSILE_FIST = Weapon("Heavy Missile-Fist", 30, 3, ap=3, unstoppable=True)

# ONI base heavy-fist replacements
ED_W_GUN_FIST = Weapon("Gun-Fist", 24, 2)
ED_W_SHRED_FIST = Weapon("Shred-Fist", 18, 2, rending=True)
ED_W_PLASMA_FIST = Weapon("Plasma-Fist", 24, 1, ap=4)
ED_W_FLAME_FIST = Weapon("Flame-Fist", 12, 1, blast=3, reliable=True)
ED_W_FUSION_FIST = Weapon("Fusion-Fist", 12, 1, ap=4, deadly=3)
ED_W_ROCKET_FIST = Weapon("Rocket-Fist", 24, 1, blast=3, indirect=True)
ED_W_MISSILE_FIST = Weapon("Missile-Fist", 30, 2, ap=3, unstoppable=True)

# Royal swords / sickles
ED_W_ROYAL_SWORD = Weapon("Royal Sword", 0, 2, ap=1, rending=True, melee=True)
ED_W_ROYAL_SICKLE = Weapon("Royal Sickle", 0, 2, ap=2, melee=True)
ED_W_HOOK_SWORD = Weapon("Hook Sword", 0, 1, ap=2, deadly=3, melee=True)
ED_W_CCW_A1 = W_CCW_A1
ED_W_CCW_A2 = W_CCW_A2
ED_W_SWORD_A1 = Weapon("Sword", 0, 1, ap=1, rending=True, melee=True)
ED_W_SWORD_A2 = Weapon("Sword", 0, 2, ap=1, rending=True, melee=True)
ED_W_SICKLE_A1 = Weapon("Sickle", 0, 1, ap=2, melee=True)
ED_W_SICKLE_A2 = Weapon("Sickle", 0, 2, ap=2, melee=True)
ED_W_MARTIAL_ARTS = Weapon("Martial Arts", 0, 1, melee=True)

# Spears (Royal Guard)
ED_W_SPEAR_SHOT = Weapon("Spear-Shot", 12, 1)
ED_W_SPEAR = Weapon("Spear", 0, 2, ap=1, melee=True)
ED_W_SPEAR_FLAME = Weapon("Spear-Flame", 12, 1, blast=3, limited=True, reliable=True)
ED_W_SPEAR_PLASMA = Weapon("Spear-Plasma", 24, 1, ap=4, limited=True)
ED_W_SPEAR_FUSE = Weapon("Spear-Fuse", 12, 1, ap=4, deadly=3, limited=True)
ED_W_SPEAR_SHRED = Weapon("Spear-Shred", 9, 3, rending=True)

# Cyber units
ED_W_TOXIN_CLAW = Weapon("Toxin Claws", 0, 2, bane=True, melee=True)
ED_W_SWARM_ATTACK = Weapon("Swarm Attacks", 0, 3, rending=True, melee=True)

# Tasers (Attack Drones)
ED_W_TASER = Weapon("Taser", 0, 1, melee=True)

# Vehicles / titans
ED_W_ARTILLERY_GUN = Weapon("Artillery Gun", 24, 4, blast=3, indirect=True)
ED_W_HEAVY_FLAMER = W_HEAVY_FLAMER
ED_W_HEAVY_ANTI_TANK_CANNON = Weapon("Heavy Anti-Tank Cannon", 30, 2, ap=3, deadly=6)
ED_W_HEAVY_BATTLE_CANNON = Weapon("Heavy Battle Cannon", 30, 4, ap=2, blast=3)
ED_W_HEAVY_AUTOCANNON = Weapon("Heavy Autocannon", 36, 9, ap=2)
ED_W_HEAVY_BURST_AUTOGUN = Weapon("Heavy Burst Auto-Gun", 18, 8, rending=True)
ED_W_ROPE_SICKLE = Weapon("Rope-Sickle", 0, 1, deadly=3, melee=True)
ED_W_ROPE_BLADE = Weapon("Rope-Blade", 0, 4, rending=True, melee=True)
ED_W_SWORD_LASER = Weapon("Sword-Laser", 12, 4, tear=True)
ED_W_HEAVY_FLAMETHROWER = Weapon("Heavy Flamethrower", 12, 4, ap=1, blast=3, reliable=True)
ED_W_ROCKET_LAUNCHER = Weapon("Rocket Launcher", 24, 4, blast=3, indirect=True)
ED_W_HEAVY_STRIKE_CANNON = Weapon("Heavy Strike Cannon", 30, 2, ap=3, deadly=6)
ED_W_HEAVY_BLAST_CANNON = Weapon("Heavy Blast Cannon", 24, 4, ap=2, blast=3)
ED_W_MISSILE_LAUNCHER_BIG = Weapon("Missile Launcher", 30, 8, ap=3, unstoppable=True)
ED_W_HEAVY_TITAN_SWORD = Weapon("Heavy Titan Sword", 0, 12, ap=2, rending=True, melee=True)
ED_W_TITAN_HEAVY_LASER_RIFLE = Weapon("Titan Heavy Laser Rifle", 24, 4, deadly=3, tear=True)
ED_W_TITAN_HEAVY_PLASMA_RIFLE = Weapon("Titan Heavy Plasma Rifle", 24, 4, ap=4, blast=3)
ED_W_TITAN_HEAVY_SHRED_RIFLE = Weapon("Titan Heavy Shred Rifle", 24, 18, rending=True)
ED_W_FIRE_TORRENT = Weapon("Fire Torrent", 18, 3, ap=1, blast=3, reliable=True)
ED_W_DRAGON_STRIKE = Weapon("Dragon Strike", 0, 12, rending=True, melee=True)
ED_W_TITAN_FISTS = Weapon("Titan Fists", 0, 12, ap=4, melee=True)
ED_W_GUIDED_MISSILES = Weapon("Guided Missiles", 30, 1, ap=2, deadly=6, unstoppable=True)
ED_W_TITAN_BLAST_CANNON = Weapon("Titan Blast Cannon", 30, 2, ap=2, blast=6, indirect=True)
ED_W_TITAN_STRIKE_CANNON = Weapon("Titan Strike Cannon", 36, 2, ap=3, deadly=6)
ED_W_ROCKET_POD = Weapon("Rocket Pod", 24, 2, blast=3, indirect=True)
ED_W_MISSILE_POD = Weapon("Missile Pod", 30, 4, ap=3, unstoppable=True)
ED_W_STOMP_A4 = Weapon("Stomp", 0, 4, ap=1, melee=True)
ED_W_STOMP_A6 = Weapon("Stomp", 0, 6, ap=2, melee=True)


# ===================================================================
# ETERNAL DYNASTY — UNIT TEMPLATES
# ===================================================================
# Implements the GF Eternal Dynasty v3.5.3 page-2 list (page 8 stars are
# excluded per user). Adjustments:
#   - Ambush keyword stripped from Ninja and Ninja Walker (units stay)
#   - Jetpack option becomes Flying-only (Ambush dropped)
#   - Caster / Casting Buff / spell upgrades dropped
#   - Ambush Beacon (Scouts upgrade) and Repel Ambushers (Hover Alert-Drone)
#     dropped — both rules only matter against Ambush
#   - Transport(N) is ignored (Dynasty APC keeps the unit)

def build_ed_unit_templates() -> list[UnitTemplate]:
    templates: list[UnitTemplate] = []

    # ---------- HEROES ----------

    # ONI Captain [1] — 95pts
    templates.append(UnitTemplate(
        "ed_oni_captain", "ONI Captain", 95, 1, quality=3, defense=3,
        tough=6, fearless=True, hero=True, faction="ed", clan_warrior=True,
        base_weapons=[ED_W_GREAT_MACE, ED_W_HEAVY_BASH],
        upgrade_slots=[
            UpgradeSlot("aura_command", "Command aura (one)", [
                UpgradeOption("captain", 10, adds_clan_warrior_boost_aura=True),
                UpgradeOption("spirit_core_bearer", 10, adds_rapid_charge_aura=True),
                UpgradeOption("sentinel", 20, adds_fearless_aura=True),
                UpgradeOption("sergeant", 40, adds_piercing_hunter_aura=True),
            ]),
            UpgradeSlot("mace_swap", "Replace Great Mace", [
                UpgradeOption("heavy_sword_shield", 30,
                              removes_weapon="Great Mace", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_SWORD],
                              adds_shielded=True),
                UpgradeOption("twin_heavy_flame_fist", 35,
                              removes_weapon="Great Mace", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_FLAME_FIST,
                                            ED_W_HEAVY_FLAME_FIST]),
                UpgradeOption("great_sword", 35,
                              removes_weapon="Great Mace", removes_count=1,
                              adds_weapons=[ED_W_GREAT_SWORD]),
                UpgradeOption("heavy_glaive", 35,
                              removes_weapon="Great Mace", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_GLAIVE]),
            ]),
            UpgradeSlot("heavy_swap", "Replace any Heavy Flame-Fist", [
                UpgradeOption("heavy_gun_fist", 5,
                              removes_weapon="Heavy Flame-Fist", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_GUN_FIST]),
                UpgradeOption("heavy_shred_fist", 5,
                              removes_weapon="Heavy Flame-Fist", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_SHRED_FIST]),
                UpgradeOption("heavy_fist_a3", 10,
                              removes_weapon="Heavy Flame-Fist", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_FIST_A3]),
                UpgradeOption("heavy_fusion_fist", 20,
                              removes_weapon="Heavy Flame-Fist", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_FUSION_FIST]),
                UpgradeOption("heavy_rocket_fist", 20,
                              removes_weapon="Heavy Flame-Fist", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_ROCKET_FIST]),
                UpgradeOption("heavy_plasma_fist", 45,
                              removes_weapon="Heavy Flame-Fist", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_PLASMA_FIST]),
                UpgradeOption("heavy_missile_fist", 50,
                              removes_weapon="Heavy Flame-Fist", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_MISSILE_FIST]),
            ]),
            UpgradeSlot("jetpack", "Jetpack (Ambush stripped: Flying only)", [
                UpgradeOption("jetpack", 45, adds_flying=True),
            ]),
            UpgradeSlot("drone", "Drone option", [
                UpgradeOption("attack_drone", 5,
                              adds_weapons=[ED_W_DRONE_LASER_GUN]),
                UpgradeOption("energy_drone", 15, adds_ignores_cover_aura=True),
            ]),
        ],
    ))

    # Dynasty Leader [1] — 50pts
    templates.append(UnitTemplate(
        "ed_dynasty_leader", "Dynasty Leader", 50, 1, quality=3, defense=4,
        tough=3, fearless=True, hero=True, faction="ed", clan_warrior=True,
        base_weapons=[ED_W_DYNASTY_PISTOL, ED_W_CCW_A2],
        upgrade_slots=[
            UpgradeSlot("aura_command", "Command aura (one)", [
                UpgradeOption("captain", 10, adds_clan_warrior_boost_aura=True),
                UpgradeOption("warlord", 15, adds_counter_attack_aura=True),
                UpgradeOption("sentinel", 20, adds_fearless_aura=True),
                UpgradeOption("sergeant", 40, adds_piercing_hunter_aura=True),
            ]),
            UpgradeSlot("ranged_swap", "Replace Dynasty Pistol and CCW", [
                UpgradeOption("twin_autogun_martial", 10,
                              removes_weapon="Dynasty Pistol", removes_count=1,
                              adds_weapons=[ED_W_LEADER_AUTOGUN, ED_W_LEADER_AUTOGUN,
                                            ED_W_MARTIAL_ARTS],
                              removes_weapon_2="CCW", removes_count_2=1),
                UpgradeOption("twin_royal_sword", 10,
                              removes_weapon="Dynasty Pistol", removes_count=1,
                              adds_weapons=[ED_W_ROYAL_SWORD, ED_W_ROYAL_SWORD],
                              removes_weapon_2="CCW", removes_count_2=1),
                UpgradeOption("twin_royal_sickle", 15,
                              removes_weapon="Dynasty Pistol", removes_count=1,
                              adds_weapons=[ED_W_ROYAL_SICKLE, ED_W_ROYAL_SICKLE],
                              removes_weapon_2="CCW", removes_count_2=1),
            ]),
            UpgradeSlot("pistol_swap", "Replace Dynasty Pistol", [
                UpgradeOption("leader_shotgun", 10,
                              removes_weapon="Dynasty Pistol", removes_count=1,
                              adds_weapons=[ED_W_LEADER_SHOTGUN]),
                UpgradeOption("leader_carbine", 10,
                              removes_weapon="Dynasty Pistol", removes_count=1,
                              adds_weapons=[ED_W_LEADER_CARBINE]),
                UpgradeOption("leader_long_rifle", 10,
                              removes_weapon="Dynasty Pistol", removes_count=1,
                              adds_weapons=[ED_W_LEADER_LONG_RIFLE]),
            ]),
            UpgradeSlot("ccw_swap", "Replace CCW", [
                UpgradeOption("royal_sickle", 10,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[ED_W_ROYAL_SICKLE]),
                UpgradeOption("royal_sword", 10,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[ED_W_ROYAL_SWORD]),
                UpgradeOption("hook_sword", 15,
                              removes_weapon="CCW", removes_count=1,
                              adds_weapons=[ED_W_HOOK_SWORD]),
            ]),
            UpgradeSlot("retinue", "Retinue option (one)", [
                # Hover Alert-Drone (Repel Ambushers) dropped per user.
                # Ninja Master keeps Stealth + Teleport (Ambush stripped).
                UpgradeOption("royal_guard_master", 20,
                              adds_regeneration=True,
                              adds_versatile_attack=True),
                UpgradeOption("ninja_master", 35,
                              adds_stealth=True, adds_teleport=True),
            ]),
        ],
    ))

    # ---------- INFANTRY ----------

    # Warriors [5] — 90pts
    templates.append(UnitTemplate(
        "ed_warriors", "Warriors", 90, 5, quality=4, defense=4,
        faction="ed", clan_warrior=True,
        base_weapons=[ED_W_LONG_RIFLE] * 5 + [ED_W_CCW_A1] * 5,
        upgrade_slots=[
            UpgradeSlot("ranged_swap_all", "Replace all Long Rifles", [
                UpgradeOption("shotguns", 15,
                              removes_weapon="Long Rifle", removes_count=5,
                              adds_weapons=[ED_W_SHOTGUN] * 5,
                              applies_to_all=True),
                UpgradeOption("carbines", 15,
                              removes_weapon="Long Rifle", removes_count=5,
                              adds_weapons=[ED_W_CARBINE] * 5,
                              applies_to_all=True),
            ]),
            UpgradeSlot("specialist", "Replace one Long Rifle", [
                UpgradeOption("flamer", 10,
                              removes_weapon="Long Rifle", removes_count=1,
                              adds_weapons=[ED_W_FLAMER]),
                UpgradeOption("shred_rifle", 10,
                              removes_weapon="Long Rifle", removes_count=1,
                              adds_weapons=[ED_W_SHRED_RIFLE]),
                UpgradeOption("plasma_rifle", 10,
                              removes_weapon="Long Rifle", removes_count=1,
                              adds_weapons=[ED_W_PLASMA_RIFLE]),
                UpgradeOption("fusion_rifle", 15,
                              removes_weapon="Long Rifle", removes_count=1,
                              adds_weapons=[ED_W_FUSION_RIFLE]),
                UpgradeOption("shishi_turret_rocket", 20,
                              removes_weapon="Long Rifle", removes_count=1,
                              adds_weapons=[ED_W_SHISHI_TURRET_ROCKET]),
                UpgradeOption("shishi_turret_missile", 30,
                              removes_weapon="Long Rifle", removes_count=1,
                              adds_weapons=[ED_W_SHISHI_TURRET_MISSILE]),
            ]),
            UpgradeSlot("sgt_loadout", "Sergeant loadout", [
                UpgradeOption("sgt_twin_pistol_ccw", 5,
                              removes_weapon="Long Rifle", removes_count=1,
                              adds_weapons=[ED_W_TWIN_HEAVY_PISTOL, ED_W_CCW_A1],
                              removes_weapon_2="CCW", removes_count_2=1),
                UpgradeOption("sgt_pistol_royal_sword", 10,
                              removes_weapon="Long Rifle", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_PISTOL, ED_W_ROYAL_SWORD],
                              removes_weapon_2="CCW", removes_count_2=1),
            ]),
        ],
    ))

    # Scouts [5] — 100pts (Scout)
    templates.append(UnitTemplate(
        "ed_scouts", "Scouts", 100, 5, quality=4, defense=5,
        scout=True, faction="ed", clan_warrior=True,
        base_weapons=[ED_W_LONG_RIFLE] * 5 + [ED_W_CCW_A1] * 5,
        upgrade_slots=[
            UpgradeSlot("ranged_swap_all", "Replace any Long Rifle", [
                UpgradeOption("carbines", 25,
                              removes_weapon="Long Rifle", removes_count=5,
                              adds_weapons=[ED_W_CARBINE] * 5,
                              applies_to_all=True),
            ]),
            UpgradeSlot("ranged_swap_3", "Replace up to three Long Rifles", [
                UpgradeOption("energy_rifle", 15,
                              removes_weapon="Long Rifle", removes_count=1,
                              adds_weapons=[ED_W_ENERGY_RIFLE]),
                UpgradeOption("sniper_rifle", 25,
                              removes_weapon="Long Rifle", removes_count=1,
                              adds_weapons=[ED_W_SNIPER_RIFLE]),
            ]),
            # Dynastic Triangulator (Ambush Beacon) dropped per user.
            UpgradeSlot("drone", "Drone (one)", [
                UpgradeOption("attack_drone", 5,
                              adds_weapons=[ED_W_DRONE_LASER_GUN]),
                UpgradeOption("energy_drone", 10, adds_ignores_cover_aura=True),
                UpgradeOption("shield_drone", 30, adds_stealth_aura=True),
            ]),
        ],
    ))

    # Ninja [5] — 165pts (Ambush stripped, Stealth + Teleport remain)
    templates.append(UnitTemplate(
        "ed_ninja", "Ninja", 165, 5, quality=3, defense=5,
        stealth=True, teleport=True, faction="ed", clan_warrior=True,
        base_weapons=[ED_W_AUTOGUN] * 5 + [ED_W_SWORD_A1] * 5,
        upgrade_slots=[
            UpgradeSlot("loadout_swap", "Replace any Auto-Gun and Sword", [
                UpgradeOption("autogun_sickle_each", 25,
                              removes_weapon="Auto-Gun", removes_count=5,
                              adds_weapons=[ED_W_AUTOGUN] * 5,
                              applies_to_all=True,
                              removes_weapon_2="Sword", removes_count_2=5,
                              adds_weapons_2=[ED_W_SICKLE_A1] * 5),
                UpgradeOption("twin_sword", 25,
                              removes_weapon="Auto-Gun", removes_count=5,
                              applies_to_all=True,
                              removes_weapon_2="Sword", removes_count_2=5,
                              adds_weapons_2=[ED_W_SWORD_A2] * 5),
                UpgradeOption("twin_sickle", 25,
                              removes_weapon="Auto-Gun", removes_count=5,
                              applies_to_all=True,
                              removes_weapon_2="Sword", removes_count_2=5,
                              adds_weapons_2=[ED_W_SICKLE_A2] * 5),
                UpgradeOption("twin_autogun_martial", 25,
                              removes_weapon="Auto-Gun", removes_count=5,
                              adds_weapons=[ED_W_AUTOGUN] * 10
                                            + [ED_W_MARTIAL_ARTS] * 5,
                              applies_to_all=True,
                              removes_weapon_2="Sword", removes_count_2=5),
            ]),
        ],
    ))

    # Royal Guard [5] — 185pts (Regeneration, Versatile Attack)
    templates.append(UnitTemplate(
        "ed_royal_guard", "Royal Guard", 185, 5, quality=3, defense=4,
        regeneration=True, versatile_attack=True,
        faction="ed", clan_warrior=True,
        base_weapons=[ED_W_SPEAR_SHOT] * 5 + [ED_W_SPEAR] * 5,
        upgrade_slots=[
            UpgradeSlot("vengeance", "Honor-Bound (Vengeance)", [
                UpgradeOption("honor_bound", 20, adds_vengeance=True),
            ]),
            UpgradeSlot("spear_swap", "Replace one Spear-Shot", [
                UpgradeOption("spear_flame", 5,
                              removes_weapon="Spear-Shot", removes_count=1,
                              adds_weapons=[ED_W_SPEAR_FLAME]),
                UpgradeOption("spear_plasma", 5,
                              removes_weapon="Spear-Shot", removes_count=1,
                              adds_weapons=[ED_W_SPEAR_PLASMA]),
                UpgradeOption("spear_fuse", 5,
                              removes_weapon="Spear-Shot", removes_count=1,
                              adds_weapons=[ED_W_SPEAR_FUSE]),
                UpgradeOption("spear_shred", 15,
                              removes_weapon="Spear-Shot", removes_count=1,
                              adds_weapons=[ED_W_SPEAR_SHRED]),
            ]),
        ],
    ))

    # Attack Drones [5] — 115pts (Strider)
    templates.append(UnitTemplate(
        "ed_attack_drones", "Attack Drones", 115, 5, quality=4, defense=4,
        strider=True, faction="ed", clan_warrior=True,
        base_weapons=[ED_W_LASER_GUN] * 5 + [ED_W_TASER] * 5,
        upgrade_slots=[
            UpgradeSlot("heavy_swap", "Replace one Laser Gun", [
                UpgradeOption("heavy_laser_gun", 30,
                              removes_weapon="Laser Gun", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_LASER_GUN]),
            ]),
        ],
    ))

    # ONIs [3] — 170pts (Tough(3))
    templates.append(UnitTemplate(
        "ed_onis", "ONIs", 170, 3, quality=3, defense=3,
        tough=3, faction="ed", clan_warrior=True,
        base_weapons=[ED_W_BASH_A1] * 3 + [ED_W_HEAVY_FIST_A1] * 6,
        upgrade_slots=[
            UpgradeSlot("vengeance", "Honor-Bound (Vengeance)", [
                UpgradeOption("honor_bound", 20, adds_vengeance=True),
            ]),
            UpgradeSlot("fist_swap_all", "Replace any Heavy Fist", [
                UpgradeOption("gun_fist", 10,
                              removes_weapon="Heavy Fist", removes_count=1,
                              adds_weapons=[ED_W_GUN_FIST]),
                UpgradeOption("shred_fist", 10,
                              removes_weapon="Heavy Fist", removes_count=1,
                              adds_weapons=[ED_W_SHRED_FIST]),
                UpgradeOption("plasma_fist", 35,
                              removes_weapon="Heavy Fist", removes_count=1,
                              adds_weapons=[ED_W_PLASMA_FIST]),
            ]),
            UpgradeSlot("fist_swap_one", "Replace one Heavy Fist", [
                UpgradeOption("flame_fist", 10,
                              removes_weapon="Heavy Fist", removes_count=1,
                              adds_weapons=[ED_W_FLAME_FIST]),
                UpgradeOption("fusion_fist", 20,
                              removes_weapon="Heavy Fist", removes_count=1,
                              adds_weapons=[ED_W_FUSION_FIST]),
                UpgradeOption("rocket_fist", 20,
                              removes_weapon="Heavy Fist", removes_count=1,
                              adds_weapons=[ED_W_ROCKET_FIST]),
                UpgradeOption("missile_fist", 40,
                              removes_weapon="Heavy Fist", removes_count=1,
                              adds_weapons=[ED_W_MISSILE_FIST]),
            ]),
            UpgradeSlot("any_2x_fist", "Any model may replace 2x Heavy Fist", [
                UpgradeOption("sword_shield", 5,
                              removes_weapon="Heavy Fist", removes_count=2,
                              adds_weapons=[ED_W_SWORD_A2],
                              adds_shielded=True),
                UpgradeOption("heavy_glaive", 5,
                              removes_weapon="Heavy Fist", removes_count=2,
                              adds_weapons=[ED_W_HEAVY_GLAIVE_A3]),
            ]),
            UpgradeSlot("jetpacks", "Jetpacks (Flying only)", [
                UpgradeOption("jetpacks", 65, adds_flying=True,
                              applies_to_all=True),
            ]),
        ],
    ))

    # Cyber-Bird Swarms [3] — 90pts (Flying, Tough(3))
    templates.append(UnitTemplate(
        "ed_cyber_bird_swarms", "Cyber-Bird Swarms", 90, 3,
        quality=5, defense=6, tough=3, flying=True,
        faction="ed", clan_warrior=True,
        base_weapons=[ED_W_SWARM_ATTACK] * 3,
        upgrade_slots=[
            UpgradeSlot("master", "Hunt Master (one)", [
                UpgradeOption("vanguard_hunt_master", 15, adds_scout=True),
                UpgradeOption("combat_hunt_master", 25,
                              adds_unpredictable_fighter=True),
            ]),
        ],
    ))

    # Cyber Lizards [5] — 90pts (Strider)
    templates.append(UnitTemplate(
        "ed_cyber_lizards", "Cyber Lizards", 90, 5, quality=4, defense=4,
        strider=True, faction="ed", clan_warrior=True,
        base_weapons=[ED_W_TOXIN_CLAW] * 5,
        upgrade_slots=[
            UpgradeSlot("master", "Hunt Master (one)", [
                UpgradeOption("vanguard_hunt_master", 15, adds_scout=True),
                UpgradeOption("combat_hunt_master", 40,
                              adds_unpredictable_fighter=True),
            ]),
        ],
    ))

    # Cyber Beast [1] — 95pts (Furious, Tough(6))
    templates.append(UnitTemplate(
        "ed_cyber_beast", "Cyber Beast", 95, 1, quality=3, defense=4,
        tough=6, furious=True, faction="ed", clan_warrior=True,
        base_weapons=[ED_W_HEAVY_FIST_A6],
        upgrade_slots=[
            UpgradeSlot("master", "Hunt Master (one)", [
                UpgradeOption("vanguard_hunt_master", 20, adds_scout=True),
                UpgradeOption("combat_hunt_master", 35,
                              adds_unpredictable_fighter=True),
            ]),
        ],
    ))

    # ---------- VEHICLES ----------

    # Dynasty APC [1] — 160pts (Fast, Impact(3), Strider, Tough(6); Transport ignored)
    templates.append(UnitTemplate(
        "ed_dynasty_apc", "Dynasty APC", 160, 1, quality=4, defense=2,
        tough=6, fast=True, impact=3, strider=True,
        faction="ed", clan_warrior=True,
        base_weapons=[ED_W_HEAVY_FLAMER],
        upgrade_slots=[
            UpgradeSlot("flamer_swap", "Replace Heavy Flamer", [
                UpgradeOption("heavy_autogun", 10,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_AUTO_GUN]),
                UpgradeOption("heavy_laser", 10,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_LASER_GUN_TANK]),
            ]),
        ],
    ))

    # Dragon Speeder [1] — 175pts (Fast, Impact(3), Strider, Tough(6))
    templates.append(UnitTemplate(
        "ed_dragon_speeder", "Dragon Speeder", 175, 1, quality=4, defense=2,
        tough=6, fast=True, impact=3, strider=True,
        faction="ed", clan_warrior=True,
        base_weapons=[ED_W_HULL_AUTOGUN, ED_W_TWIN_BURST_LASER_GUN],
        upgrade_slots=[
            UpgradeSlot("hull_swap", "Replace Hull Auto-Gun", [
                UpgradeOption("hull_flamer", 5,
                              removes_weapon="Hull Auto-Gun", removes_count=1,
                              adds_weapons=[ED_W_HULL_FLAMER]),
                UpgradeOption("hull_laser_gun", 15,
                              removes_weapon="Hull Auto-Gun", removes_count=1,
                              adds_weapons=[ED_W_HULL_LASER_GUN]),
            ]),
            UpgradeSlot("targeting_beam", "Targeting Beam (ISR Mark)", [
                UpgradeOption("targeting_beam", 35, adds_isr_mark=True),
            ]),
            UpgradeSlot("twin_burst_swap", "Replace Twin Burst Laser Gun", [
                UpgradeOption("twin_heavy_laser", 30,
                              removes_weapon="Twin Burst Laser Gun", removes_count=1,
                              adds_weapons=[ED_W_TWIN_HEAVY_LASER_GUN]),
            ]),
        ],
    ))

    # Dynasty Tank [1] — 360pts (Fast, Impact(6), Strider, Tough(12))
    templates.append(UnitTemplate(
        "ed_dynasty_tank", "Dynasty Tank", 360, 1, quality=4, defense=2,
        tough=12, fast=True, impact=6, strider=True,
        faction="ed", clan_warrior=True,
        base_weapons=[ED_W_ARTILLERY_GUN, ED_W_HEAVY_FLAMER],
        upgrade_slots=[
            UpgradeSlot("primary_swap", "Replace Artillery Gun", [
                UpgradeOption("heavy_anti_tank", 40,
                              removes_weapon="Artillery Gun", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_ANTI_TANK_CANNON]),
                UpgradeOption("heavy_battle_cannon", 55,
                              removes_weapon="Artillery Gun", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_BATTLE_CANNON]),
                UpgradeOption("heavy_autocannon", 60,
                              removes_weapon="Artillery Gun", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_AUTOCANNON]),
            ]),
            UpgradeSlot("flamer_swap", "Replace Heavy Flamer", [
                UpgradeOption("heavy_autogun", 10,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_AUTO_GUN]),
                UpgradeOption("heavy_laser_gun", 10,
                              removes_weapon="Heavy Flamer", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_LASER_GUN_TANK]),
            ]),
        ],
    ))

    # Ninja Walker [1] — 325pts (Ambush stripped; Fear(2), Stealth, Tough(9))
    templates.append(UnitTemplate(
        "ed_ninja_walker", "Ninja Walker", 325, 1, quality=3, defense=2,
        tough=9, fear=2, stealth=True, faction="ed", clan_warrior=True,
        base_weapons=[ED_W_HEAVY_BURST_AUTOGUN, ED_W_ROPE_SICKLE, ED_W_STOMP_A4],
        upgrade_slots=[
            UpgradeSlot("targeting_beam", "Targeting Beam (ISR Mark)", [
                UpgradeOption("targeting_beam", 35, adds_isr_mark=True),
            ]),
            UpgradeSlot("burst_swap", "Replace Heavy Burst Auto-Gun", [
                UpgradeOption("sword_laser_combo", 10,
                              removes_weapon="Heavy Burst Auto-Gun", removes_count=1,
                              adds_weapons=[ED_W_SWORD_LASER, ED_W_HEAVY_SWORD_A6]),
            ]),
            UpgradeSlot("rope_swap", "Replace Rope-Sickle", [
                UpgradeOption("rope_blade", 15,
                              removes_weapon="Rope-Sickle", removes_count=1,
                              adds_weapons=[ED_W_ROPE_BLADE]),
                UpgradeOption("displacement_pack", 25,
                              adds_ed_teleport=True),
            ]),
        ],
    ))

    # ONI Walker [1] — 320pts (Fear(2), Tough(12))
    templates.append(UnitTemplate(
        "ed_oni_walker", "ONI Walker", 320, 1, quality=3, defense=2,
        tough=12, fear=2, faction="ed", clan_warrior=True,
        base_weapons=[ED_W_HEAVY_FLAMETHROWER, ED_W_STOMP_A4],
        upgrade_slots=[
            UpgradeSlot("flame_swap", "Replace Heavy Flamethrower", [
                UpgradeOption("rocket_launcher", 25,
                              removes_weapon="Heavy Flamethrower", removes_count=1,
                              adds_weapons=[ED_W_ROCKET_LAUNCHER]),
                UpgradeOption("heavy_strike_cannon", 75,
                              removes_weapon="Heavy Flamethrower", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_STRIKE_CANNON]),
                UpgradeOption("heavy_blast_cannon", 75,
                              removes_weapon="Heavy Flamethrower", removes_count=1,
                              adds_weapons=[ED_W_HEAVY_BLAST_CANNON]),
                UpgradeOption("missile_launcher_big", 100,
                              removes_weapon="Heavy Flamethrower", removes_count=1,
                              adds_weapons=[ED_W_MISSILE_LAUNCHER_BIG]),
            ]),
            UpgradeSlot("drone", "Drone (one)", [
                UpgradeOption("attack_drone_laser", 15,
                              adds_weapons=[Weapon("Drone Laser Gun (always)", 18, 1, tear=True)]),
                UpgradeOption("shield_drone", 40, adds_stealth=True),
            ]),
        ],
    ))

    # ---------- TITANS ----------

    # Samurai Titan [1] — 515pts
    templates.append(UnitTemplate(
        "ed_samurai_titan", "Samurai Titan", 515, 1, quality=3, defense=2,
        tough=18, fear=3, fearless=True, faction="ed", clan_warrior=True,
        base_weapons=[ED_W_HEAVY_TITAN_SWORD, ED_W_STOMP_A6],
        upgrade_slots=[
            UpgradeSlot("sword_swap", "Replace Heavy Titan Sword", [
                UpgradeOption("titan_heavy_laser_rifle", 75,
                              removes_weapon="Heavy Titan Sword", removes_count=1,
                              adds_weapons=[ED_W_TITAN_HEAVY_LASER_RIFLE]),
                UpgradeOption("titan_heavy_plasma_rifle", 100,
                              removes_weapon="Heavy Titan Sword", removes_count=1,
                              adds_weapons=[ED_W_TITAN_HEAVY_PLASMA_RIFLE]),
                UpgradeOption("titan_heavy_shred_rifle", 130,
                              removes_weapon="Heavy Titan Sword", removes_count=1,
                              adds_weapons=[ED_W_TITAN_HEAVY_SHRED_RIFLE]),
            ]),
            UpgradeSlot("force_field", "Force Field (Stealth)", [
                UpgradeOption("force_field", 65, adds_stealth=True),
            ]),
        ],
    ))

    # Macaque Titan [1] — 650pts (Bounding, Fear(3), Fearless, Furious, Strider)
    templates.append(UnitTemplate(
        "ed_macaque_titan", "Macaque Titan", 650, 1, quality=3, defense=2,
        tough=18, fear=3, fearless=True, furious=True, strider=True,
        faction="ed", clan_warrior=True, bounding=True,
        base_weapons=[ED_W_STOMP_A6, ED_W_TITAN_FISTS],
    ))

    # Dragon Titan [1] — 815pts (Caster(4) dropped per spell-skip;
    # Fear(3), Fearless, Flying, Regeneration retained)
    templates.append(UnitTemplate(
        "ed_dragon_titan", "Dragon Titan", 815, 1, quality=3, defense=2,
        tough=18, fear=3, fearless=True, flying=True, regeneration=True,
        faction="ed", clan_warrior=True,
        base_weapons=[ED_W_FIRE_TORRENT, ED_W_DRAGON_STRIKE, ED_W_STOMP_A6],
    ))

    # Artillery Titan [1] — 735pts (Slow, Fear(3), Fearless)
    templates.append(UnitTemplate(
        "ed_artillery_titan", "Artillery Titan", 735, 1, quality=3, defense=2,
        tough=18, fear=3, fearless=True, slow=True,
        faction="ed", clan_warrior=True,
        base_weapons=[ED_W_GUIDED_MISSILES, ED_W_TITAN_BLAST_CANNON,
                      ED_W_ROCKET_POD, ED_W_STOMP_A6],
        upgrade_slots=[
            UpgradeSlot("blast_cannon_swap", "Replace Titan Blast Cannon", [
                UpgradeOption("titan_strike_cannon", 10,
                              removes_weapon="Titan Blast Cannon", removes_count=1,
                              adds_weapons=[ED_W_TITAN_STRIKE_CANNON]),
            ]),
            UpgradeSlot("rocket_pod_swap", "Replace Rocket Pod", [
                UpgradeOption("missile_pod", 35,
                              removes_weapon="Rocket Pod", removes_count=1,
                              adds_weapons=[ED_W_MISSILE_POD]),
            ]),
        ],
    ))

    return templates


# ===================================================================
# COMBINED TEMPLATE GENERATION
# ===================================================================

def _generate_combined_templates(
    base_templates: list[UnitTemplate],
) -> list[UnitTemplate]:
    """Auto-generate combined (doubled) variants for eligible templates.

    Eligible: non-hero, multi-model (size > 1).
    """
    import copy as _copy

    combined: list[UnitTemplate] = []
    for tpl in base_templates:
        if tpl.hero or tpl.size <= 1:
            continue

        # Classify slots: shared (all options applies_to_all) vs per-half
        shared_slots: list[UpgradeSlot] = []
        per_half_slots: list[UpgradeSlot] = []
        for slot in tpl.upgrade_slots:
            if slot.options and all(o.applies_to_all for o in slot.options):
                shared_slots.append(slot)
            else:
                per_half_slots.append(slot)

        # Build upgrade slots for the combined template
        new_slots: list[UpgradeSlot] = []

        # Shared slots: same ID, doubled cost
        for slot in shared_slots:
            new_opts = []
            for opt in slot.options:
                o = _copy.copy(opt)
                o.cost = opt.cost * 2
                new_opts.append(o)
            new_slots.append(UpgradeSlot(
                id=slot.id,
                description=slot.description,
                options=new_opts,
            ))

        # Per-half slots: duplicated with _a and _b suffixes
        for slot in per_half_slots:
            for suffix in ("_a", "_b"):
                new_slots.append(UpgradeSlot(
                    id=slot.id + suffix,
                    description=slot.description,
                    options=list(slot.options),  # same options, same costs
                ))

        combined.append(UnitTemplate(
            id=f"{tpl.id}_combined",
            name=f"{tpl.name} (Combined)",
            base_cost=tpl.base_cost * 2,
            size=tpl.size * 2,
            quality=tpl.quality,
            defense=tpl.defense,
            tough=tpl.tough,
            fearless=tpl.fearless,
            regeneration=tpl.regeneration,
            base_weapons=[w for i in range(0, len(tpl.base_weapons), tpl.size)
                         for w in list(tpl.base_weapons[i:i + tpl.size]) * 2],
            upgrade_slots=new_slots,
            scout=tpl.scout,
            stealth=tpl.stealth,
            relentless=tpl.relentless,
            fast=tpl.fast,
            artillery=tpl.artillery,
            shielded=tpl.shielded,
            furious=tpl.furious,
            impact=tpl.impact,
            fortified=tpl.fortified,
            hero=False,
            flying=tpl.flying,
            teleport=tpl.teleport,
            fear=tpl.fear,
            is_combined=True,
            source_template_id=tpl.id,
            faction=tpl.faction,
            battleborn=tpl.battleborn,
            strider=tpl.strider,
            versatile_attack=tpl.versatile_attack,
            versatile_reach=tpl.versatile_reach,
            unstoppable_mark=tpl.unstoppable_mark,
            clan_warrior=tpl.clan_warrior,
            clan_warrior_boost=tpl.clan_warrior_boost,
            piercing_hunter=tpl.piercing_hunter,
            melee_evasion=tpl.melee_evasion,
            counter_attack=tpl.counter_attack,
            unpredictable_fighter=tpl.unpredictable_fighter,
            rapid_advance=tpl.rapid_advance,
            rapid_charge=tpl.rapid_charge,
            bounding=tpl.bounding,
            ed_teleport=tpl.ed_teleport,
            vengeance=tpl.vengeance,
            isr_mark=tpl.isr_mark,
            ignores_cover=tpl.ignores_cover,
            slow=tpl.slow,
            precision_fighter=tpl.precision_fighter,
        ))

    return combined


# ===================================================================
# TEMPLATE CACHE
# ===================================================================

_TEMPLATES: list[UnitTemplate] | None = None
_TEMPLATES_DICT: dict[str, UnitTemplate] | None = None


def get_templates() -> list[UnitTemplate]:
    global _TEMPLATES
    if _TEMPLATES is None:
        hef_base = build_unit_templates()
        bb_base = build_bb_unit_templates()
        ed_base = build_ed_unit_templates()
        base = hef_base + bb_base + ed_base
        _TEMPLATES = base + _generate_combined_templates(base)
    return _TEMPLATES


def get_templates_by_faction(faction: str) -> list[UnitTemplate]:
    """Return templates for one faction ('hef' or 'bb'). Includes combined variants."""
    return [t for t in get_templates() if t.faction == faction]


def get_templates_dict() -> dict[str, UnitTemplate]:
    global _TEMPLATES_DICT
    if _TEMPLATES_DICT is None:
        _TEMPLATES_DICT = {t.id: t for t in get_templates()}
    return _TEMPLATES_DICT
