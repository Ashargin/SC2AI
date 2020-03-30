# Unit hidden attributes : all types are np.int64
UNIT_ATTRS = {'_index_names': [
    {'unit_type': 0,
    'alliance': 1,
    'health': 2,
    'shield': 3,
    'energy': 4,
    'cargo_space_taken': 5,
    'build_progress': 6,
    'health_ratio': 7,
    'shield_ratio': 8,
    'energy_ratio': 9,
    'display_type': 10,
    'owner': 11,
    'x': 12,
    'y': 13,
    'facing': 14,
    'radius': 15,
    'cloak': 16,
    'is_selected': 17,
    'is_blip': 18,
    'is_powered': 19,
    'mineral_contents': 20,
    'vespene_contents': 21,
    'cargo_space_max': 22,
    'assigned_harvesters': 23,
    'ideal_harvesters': 24,
    'weapon_cooldown': 25,
    'order_length': 26,
    'order_id_0': 27,
    'order_id_1': 28,
    'tag': 29,
    'hallucination': 30,
    'buff_id_0': 31,
    'buff_id_1': 32,
    'addon_unit_type': 33,
    'active': 34,
    'is_on_screen': 35,
    'order_progress_0': 36,
    'order_progress_1': 37,
    'order_id_2': 38,
    'order_id_3': 39,
    'is_in_cargo': 40,
    'buff_duration_remain': 41,
    'buff_duration_max': 42,
    'attack_upgrade_level': 43,
    'armor_upgrade_level': 44,
    'shield_upgrade_level': 45}
    ]}

# Drone example :
# 104 unit_type
# 1 alliance
# 40 health
# 0 shield
# 0 energy
# 0 cargo_space_taken
# 100 build_progress
# 255 health_ratio
# 0 shield_ratio
# 0 energy_ratio
# 1 display_type
# 1 owner
# 37 x
# 56 y
# 2 facing
# 1 radius
# 3 cloak
# 0 is_selected
# 0 is_blip
# 0 is_powered
# 0 mineral_contents
# 0 vespene_contents
# 0 cargo_space_max
# 0 assigned_harvesters
# 0 ideal_harvesters
# 0 weapon_cooldown
# 1 order_length
# 356 order_id_0
# 0 order_id_1
# 0 tag
# 0 hallucination
# 0 buff_id_0
# 0 buff_id_1
# 0 addon_unit_type
# 1 active
# 1 is_on_screen
# 0 order_progress_0
# 0 order_progress_1
# 0 order_id_2
# 0 order_id_3
# 0 is_in_cargo
# 0 buff_duration_remain
# 0 buff_duration_max
# 0 attack_upgrade_level
# 0 armor_upgrade_level
# 0 shield_upgrade_level

# Extractor example :
# 88 unit_type
# 1 alliance
# 500 health
# 0 shield
# 0 energy
# 0 cargo_space_taken
# 100 build_progress
# 255 health_ratio
# 0 shield_ratio
# 0 energy_ratio
# 1 display_type
# 1 owner
# 25 x
# 56 y
# 4 facing
# 5 radius
# 3 cloak
# 0 is_selected
# 0 is_blip
# 0 is_powered
# 0 mineral_contents
# 2250 vespene_contents
# 0 cargo_space_max
# 0 assigned_harvesters
# 3 ideal_harvesters
# 0 weapon_cooldown
# 0 order_length
# 0 order_id_0
# 0 order_id_1
# 0 tag
# 0 hallucination
# 0 buff_id_0
# 0 buff_id_1
# 0 addon_unit_type
# 0 active
# 1 is_on_screen
# 0 order_progress_0
# 0 order_progress_1
# 0 order_id_2
# 0 order_id_3
# 0 is_in_cargo
# 0 buff_duration_remain
# 0 buff_duration_max
# 0 attack_upgrade_level
# 0 armor_upgrade_level
# 0 shield_upgrade_level

# Hatchery example :
# 86 unit_type
# 1 alliance
# 1500 health
# 0 shield
# 0 energy
# 0 cargo_space_taken
# 100 build_progress
# 255 health_ratio
# 0 shield_ratio
# 0 energy_ratio
# 1 display_type
# 1 owner
# 76 x
# 64 y
# 4 facing
# 9 radius
# 3 cloak
# 0 is_selected
# 0 is_blip
# 0 is_powered
# 0 mineral_contents
# 0 vespene_contents
# 0 cargo_space_max
# 0 assigned_harvesters
# 0 ideal_harvesters
# 0 weapon_cooldown
# 0 order_length
# 0 order_id_0
# 0 order_id_1
# 0 tag
# 0 hallucination
# 0 buff_id_0
# 0 buff_id_1
# 0 addon_unit_type
# 0 active
# 1 is_on_screen
# 0 order_progress_0
# 0 order_progress_1
# 0 order_id_2
# 0 order_id_3
# 0 is_in_cargo
# 0 buff_duration_remain
# 0 buff_duration_max
# 0 attack_upgrade_level
# 0 armor_upgrade_level
# 0 shield_upgrade_level

# obs attributes :
OBS_ATTRS = ['count', 'discount', 'first', 'index', 'last', 'mid', 'observation', 'reward', 'step_type']

# obs.observation attributes :
OBS_OBS_ATTRS = ['action_result', 'alerts', 'available_actions', 'away_race_requested', 'build_queue', 'camera_position', 'camera_size', 'cargo', 'cargo_slots_available', 'clear', 'control_groups', 'copy', 'feature_effects', 'feature_minimap', 'feature_screen', 'feature_units', 'fromkeys', 'game_loop', 'get', 'home_race_requested', 'items', 'keys', 'last_actions', 'map_name', 'multi_select', 'player', 'pop', 'popitem', 'production_queue', 'radar', 'raw_effects', 'raw_units', 'score_by_category', 'score_by_vital', 'score_cumulative', 'setdefault', 'single_select', 'unit_counts', 'update', 'upgrades', 'values']
