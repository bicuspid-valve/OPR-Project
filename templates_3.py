"""Unit templates and weapon constants for High Elf Fleets."""
from __future__ import annotations

from models import Weapon, UpgradeOption, UpgradeSlot, UnitTemplate


# ===================================================================
# WEAPON CONSTANTS
# ===================================================================

W_SHARDGUN = Weapon("Shardgun", 12, 2, crack=True)
W_SHARD_PISTOL = Weapon("Shard Pistol", 12, 1, crack=True)
W_SHARD_CARBINE = Weapon("Shard Carbine", 18, 2, crack=True, deadly = 3)
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
                              adds_weapons=[Weapon("Twin Shard Carbine", 18, 4, crack=True, deadly=3)],
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
                              adds_weapons=[Weapon("Master Shard Carbine", 18, 3, crack=True, deadly=3)],
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
                              adds_weapons=[Weapon("Master Shard Carbine", 18, 3, crack=True, deadly=3)],
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
        base = build_unit_templates()
        _TEMPLATES = base + _generate_combined_templates(base)
    return _TEMPLATES


def get_templates_dict() -> dict[str, UnitTemplate]:
    global _TEMPLATES_DICT
    if _TEMPLATES_DICT is None:
        _TEMPLATES_DICT = {t.id: t for t in get_templates()}
    return _TEMPLATES_DICT
