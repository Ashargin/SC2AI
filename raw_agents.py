import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

import numpy as np
import random as rd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from pysc2.lib import actions, features, buffs, units
from raw_base_agent import ZergAgent
from settings import STEP_MUL


class ZerglingRush(ZergAgent):
    """Build : https://lotv.spawningtool.com/build/119713/"""

    def __init__(self):
        super().__init__()
        self.mode = 'build'
        self.zerglings_to_push = 12
        self.attack_ended = False

    def reset(self):
        super().reset()
        self.mode = 'build'
        self.attack_ended = False

    def update_mode(self):
        
        zerglings = [u for u in self.get_units_agg('Zergling')
                       if np.abs(np.array(self.enemy_coordinates) - (u.x, u.y)).sum()
                       <= np.abs(np.array(self.self_coordinates) - (u.x, u.y)).sum()]

        if self.mode == 'build' and len(zerglings) >= 2 + self.zerglings_to_push:
            self.mode = 'attack'
        if self.mode == 'attack' and len(zerglings) <= int((2 + self.zerglings_to_push) / 2):
            self.mode = 'build'

    def attack_then_hunt(self, unit_names='all', can_reach_ground=True, can_reach_air=True):
        if self.attack_ended:
            self.hunt_hl(can_reach_ground=can_reach_ground, can_reach_air=can_reach_air,
                                                unit_names=unit_names, where='median')
        else:
            self.attack_hl(can_reach_ground=can_reach_ground, can_reach_air=can_reach_air,
                                                unit_names=unit_names, where='head')
            if self.action is None or self.action.function == actions.RAW_FUNCTIONS.no_op.id:
                self.attack_ended = True

    def step(self, obs):
        super().step(obs)

        self.update_mode()

        # Build order
        # Always needed
        self.scout_overlord_hl(coordinates='enemy_close', timeout=np.inf)
        self.cast_inject_larva_hl()
        self.set_rally_point_units_hl(coordinates='proxy')

        # Building mode
        if self.mode == 'build':
            self.build_spawning_pool_hl()
            self.train_zergling_hl(6, pop=14)
            self.build_hatchery_hl(2)
            self.train_zergling_hl(10)
            self.train_queen_hl(1)
            self.build_spine_crawler_hl(1, pop=22, increase=self.train_zergling)
            self.train_zergling_hl(np.inf)
            # Move troops back if the attack is canceled:
            self.move_hl(coordinates='proxy')

        # Attack mode
        elif self.mode == 'attack':
            self.train_zergling_hl(np.inf)
            self.attack_then_hunt()

        self.wait_hl()

        return self.action


class MacroZerg(ZergAgent):
    """Inspired from pro player games"""

    def __init__(self):
        super().__init__()
        self.finished_opening = False
        self.finished_followup = False
        self.finished_midgame = False
        self.midgame_increase = None
        self.attack_ended = False

    def reset(self):
        super().reset()
        self.finished_opening = False
        self.finished_followup = False
        self.finished_midgame = False
        self.midgame_increase = None
        self.attack_ended = False

    def essentials(self):
        self.cast_inject_larva_hl()
        self.move_workers_to_gas_hl(timeout=150)
        self.adjust_workers_distribution_intra_hl(timeout=150)
        self.adjust_workers_distribution_inter_hl(timeout=150)
        self.set_rally_point_units_hl(coordinates='home')
        self.scout_overlord_hl(coordinates='enemy_close', timeout=3000)
        if self.game_step % 10 == 0:
            self.defend_hl(defend_up_to=0.4, with_queen=False)
        self.build_creep_tumor_queen_hl(10)
        self.build_creep_tumor_tumor_hl()

        pop = self.obs.observation.player.food_used
        cap = self.obs.observation.player.food_cap
        if cap >= 66 and pop >= cap - 10 and cap < 200:
            self.train_overlord_hl(self.unit_count_agg('Overlord', with_training=False) + 2)

    def opening(self):
        if self.finished_opening:
            return

        self.train_overlord_hl(2, pop=13)
        self.build_hatchery_hl(2, pop=16)
        self.build_extractor_hl(1, how=1, pop=18)
        self.build_spawning_pool_hl(pop=17)
        self.train_overlord_hl(3, pop=20)
        self.train_queen_hl(2, pop=20)
        self.train_zergling_hl(4, pop=20)
        self.research_metabolic_boost_hl(pop=26)
        self.build_hatchery_hl(3, pop=30)
        self.train_queen_hl(3, pop=30)
        self.train_overlord_hl(4, pop=32)
        self.research_pneumatized_carapace_hl(pop=36)

        if self.obs.observation.player.food_used >= 36 and self.action is None:
            self.finished_opening = True

    def followup_standard(self):
        if self.finished_followup:
            return

        self.train_zergling_hl(6, pop=35)
        self.train_overlord_hl(5, pop=35)
        self.train_queen_hl(4, pop=39)
        self.train_zergling_hl(8, pop=43)
        self.build_spore_crawler_hl(1, how=2, pop=43)
        self.train_overlord_hl(6, pop=43)
        self.train_zergling_hl(10, pop=43)
        self.train_queen_hl(5, pop=48)
        self.train_zergling_hl(12, pop=51)
        self.build_spore_crawler_hl(2, how=1, pop=57)
        self.build_lair_hl(pop=61)

        if self.obs.observation.player.food_used >= 61 and self.action is None:
            self.finished_followup = True

    def followup_eco(self):
        if self.finished_followup:
            return

        self.build_spore_crawler_hl(1, how=2, pop=30)
        self.train_overlord_hl(5, pop=40)
        self.train_queen_hl(4, pop=42)
        self.train_overlord_hl(6, pop=49)
        self.train_queen_hl(5, pop=56)
        self.build_lair_hl(pop=58)

        if self.obs.observation.player.food_used >= 58 and self.action is None:
            self.finished_followup = True

    def midgame_roach_hydralisk(self):
        def increase():
            if self.unit_count_agg('Drone') < 70:
                return self.train_drone()
            elif self.obs.observation.player.vespene < 150:
                return self.train_zergling()
            self.train_roach_hl(20)
            self.train_hydralisk_hl(np.inf)
            return self.action if self.action is not None else self.wait()

        self.midgame_increase = increase
        if self.finished_midgame:
            return

        self.train_queen_hl(6, pop=58, increase=increase)
        self.build_roach_warren_hl(pop=64, increase=increase)
        self.train_queen_hl(7, pop=64, increase=increase)
        self.build_spore_crawler_hl(1, how=2, pop=73, increase=increase)
        self.build_spore_crawler_hl(2, how=1, pop=73, increase=increase)
        self.build_spore_crawler_hl(3, how=3, pop=73, increase=increase)
        self.build_extractor_hl(2, how=1, pop=76, increase=increase)
        self.build_extractor_hl(3, how=2, pop=76, increase=increase)
        self.train_roach_hl(5, pop=76, increase=increase)
        self.train_overseer_hl(1, pop=76, increase=increase)
        self.build_evolution_chamber_hl(1, pop=85, increase=increase)
        self.build_spore_crawler_hl(4, how=2, pop=91, increase=increase)
        self.build_extractor_hl(4, how=2, pop=85, increase=increase)
        self.build_extractor_hl(5, how=3, pop=92, increase=increase)
        self.research_glial_reconstitution_hl(pop=92, increase=increase)
        self.research_missile_attack_1_hl(pop=92, increase=increase)
        self.build_hatchery_hl(4, pop=92, increase=increase)
        self.build_extractor_hl(6, how=3, pop=92, increase=increase)
        self.build_infestation_pit_hl(pop=96, increase=increase)
        self.build_hydralisk_den_hl(pop=96, increase=increase)
        self.train_roach_hl(10, pop=96, increase=increase)
        self.build_spore_crawler_hl(5, how=1, pop=97, increase=increase)
        self.research_grooved_spines_hl(pop=117, increase=increase)
        self.build_spore_crawler_hl(6, how=3, pop=118, increase=increase)
        self.train_queen_hl(8, pop=118, increase=increase)
        self.build_hive_hl(pop=118, increase=increase)
        self.train_hydralisk_hl(10, pop=118, increase=increase)
        self.build_hatchery_hl(5, pop=129, increase=increase)
        self.build_extractor_hl(8, how=5, pop=129, increase=increase)
        self.research_muscular_augments_hl(pop=153, increase=increase)
        self.research_missile_attack_2_hl(pop=159, increase=increase)
        self.build_hatchery_hl(6, pop=159, increase=increase)
        self.build_extractor_hl(10, how=6, pop=159, increase=increase)
        self.build_evolution_chamber_hl(2, pop=159, increase=increase)
        self.train_overseer_hl(2, pop=158, increase=increase)
        self.research_ground_armor_1_hl(pop=158, increase=increase)

        if self.obs.observation.player.food_used >= 158 and self.action is None:
            self.finished_midgame = True

    def midgame_baneling(self):
        def increase():
            if self.unit_count_agg('Drone') < 70:
                return self.train_drone()
            elif self.obs.observation.player.vespene < 150:
                return self.train_zergling()
            self.train_hydralisk_hl(14)
            self.train_baneling_hl(np.inf)
            return self.action if self.action is not None else self.wait()

        self.midgame_increase = increase
        if self.finished_midgame:
            return

        self.build_baneling_nest_hl(pop=58, increase=increase)
        self.train_queen_hl(6, pop=62, increase=increase)
        self.train_zergling_hl(10, pop=62, increase=increase)
        self.build_extractor_hl(2, how=1, pop=62, increase=increase)
        self.build_extractor_hl(3, how=2, pop=62, increase=increase)
        self.train_zergling_hl(12, pop=72, increase=increase)
        self.build_hatchery_hl(4, pop=77, increase=increase)
        self.build_extractor_hl(4, how=2, pop=62, increase=increase)
        self.build_extractor_hl(5, how=3, pop=62, increase=increase)
        self.train_overseer_hl(1, pop=78, increase=increase)
        self.build_evolution_chamber_hl(2, pop=78, increase=increase)
        self.research_centrifugal_hooks_hl(pop=80, increase=increase)
        self.train_baneling_hl(8, pop=81, increase=increase)
        self.research_melee_attack_1_hl(pop=94, increase=increase)
        self.build_hydralisk_den_hl(pop=94, increase=increase)
        self.build_hatchery_hl(5, pop=93, increase=increase)
        self.train_queen_hl(7, pop=98, increase=increase)
        self.train_baneling_hl(10, pop=105, increase=increase)
        self.research_muscular_augments_hl(pop=109, increase=increase)
        self.build_extractor_hl(6, how=3, pop=109, increase=increase)
        self.build_infestation_pit_hl(pop=109, increase=increase)
        self.research_ground_armor_1_hl(pop=115, increase=increase)
        self.build_hatchery_hl(6, pop=119, increase=increase)
        self.build_hive_hl(pop=118, increase=increase)
        self.train_hydralisk_hl(5, pop=127, increase=increase)
        self.build_extractor_hl(7, how=5, pop=137, increase=increase)
        self.train_hydralisk_hl(10, pop=140, increase=increase)
        self.research_grooved_spines_hl(pop=140, increase=increase)
        self.research_melee_attack_2_hl(pop=147, increase=increase)
        self.build_extractor_hl(8, how=5, pop=150, increase=increase)

        if self.obs.observation.player.food_used >= 150 and self.action is None:
            self.finished_midgame = True

    def lategame_ultralisk_broodlord(self):
        def increase():
            if self.unit_count_agg('UltraliskCavern', with_training=False) == 0 \
                        or self.unit_count_agg('GreaterSpire', with_training=False) == 0:
                return self.midgame_increase()
            elif self.unit_count_agg('Drone') < 70:
                return self.train_drone()
            elif self.obs.observation.player.vespene < 300:
                return self.train_zergling()
            self.train_ultralisk_hl(2)
            self.train_brood_lord_hl(2)
            self.train_corruptor_hl(3)
            self.train_ultralisk_hl(4)
            self.train_brood_lord_hl(4)
            self.train_corruptor_hl(6)
            self.train_ultralisk_hl(6)
            self.train_brood_lord_hl(6)
            self.train_corruptor_hl(9)
            self.train_brood_lord_hl(np.inf)
            return self.action if self.action is not None else self.wait()

        self.build_spire_hl(pop=140, increase=increase)
        self.build_greater_spire_hl(pop=140, increase=increase)
        self.build_ultralisk_cavern_hl(pop=140, increase=increase)
        self.research_chitinous_plating_hl(pop=160, increase=increase)
        self.research_ground_armor_2_hl(pop=160, increase=increase)
        self.research_flyer_attack_1_hl(pop=160, increase=increase)
        self.research_melee_attack_1_hl(pop=160, increase=increase)
        self.build_spire_hl(2, pop=170, increase=increase)
        self.research_flyer_armor_1_hl(pop=170, increase=increase)
        self.research_flyer_attack_2_hl(pop=170, increase=increase)
        self.research_anabolic_synthesis_hl(pop=170, increase=increase)
        self.research_adrenal_glands_hl(pop=180, increase=increase)
        self.research_melee_attack_2_hl(pop=180, increase=increase)
        self.research_flyer_armor_2_hl(pop=180, increase=increase)
        self.research_flyer_attack_3_hl(pop=180, increase=increase)
        self.research_flyer_armor_3_hl(pop=180, increase=increase)
        self.research_melee_attack_3_hl(pop=180, increase=increase)
        self.wait_hl(pop=200, increase=increase)

        if self.obs.observation.player.food_used >= 190:
            if self.game_step % 2 == 0 or self.unit_count_agg('Corruptor', with_training=False) == 0:
                ZerglingRush.attack_then_hunt(self, can_reach_air=False,
                                              unit_names=['Zergling', 'Baneling',
                                                          'Roach', 'Hydralisk',
                                                          'Ultralisk', 'BroodLord'])
            else:
                ZerglingRush.attack_then_hunt(self, can_reach_ground=False,
                                              unit_names=['Corruptor'])

    def step(self, obs):
        super().step(obs)

        # Build order
        # Always needed
        self.essentials()
        self.opening()
        self.followup_standard()
        self.midgame_baneling()
        self.lategame_ultralisk_broodlord()

        self.wait_hl()
        return self.action

        # Building mode
        ### harass / scout with zerglings

        ### add prios
        ### issue attack frozen
        ### hatchery may be stuck
        ### supply block
        ### occasional pb building 4th base
        ### missing lings? add army
        ### add drone count
        ### add overlords
        ### move overlords to base

class SmartZerg(ZergAgent):
    def __init__(self, learning_rate=0.001, discount=0.99, batch_size=128, memory_size=1024,
                 epsilon=1.0, epsilon_decay=0.9999, epsilon_min=0.1, episode_length=25000,
                 action_space_mode='full', observation_space_mode='full'):
        super().__init__()
        self.learning_rate = learning_rate
        self.discount = discount
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episode_length = episode_length
        self.action_space_mode = action_space_mode
        self.observation_space_mode = observation_space_mode
        self.action_space = self.get_action_space()
        self.net = None
        self.position = 0
        self.previous_state = None
        self.previous_q_values = None
        self.previous_action = None
        self.replay_memory = []
        self.rewards = []
        self.attack_ended = False

    def reset(self):
        super().reset()
        self.position = 0
        self.previous_state = None
        self.previous_q_values = None
        self.previous_action = None
        self.replay_memory = []
        self.rewards.append([])
        self.attack_ended = False

    def get_action_space(self):
        if self.action_space_mode == 'limited':
            return [self.train_drone,
                    self.train_queen,
                    self.train_roach,
                    self.train_zergling,
                    self.train_hydralisk,
                    self.research_missile_attack_2,
                    self.build_hatchery,
                    self.scout_zergling,
                    self.defend,
                    self.wait]

        return [self.train_baneling,
                self.train_brood_lord,
                self.train_corruptor,
                self.train_drone,
                self.train_hydralisk,
                self.train_mutalisk,
                self.train_overlord,
                self.train_overseer,
                self.train_queen,
                self.train_ravager,
                self.train_roach,
                self.train_ultralisk,
                self.train_zergling,
                self.build_creep_tumor_queen,
                self.build_creep_tumor_tumor,
                self.build_evolution_chamber,
                self.build_extractor,
                self.build_hatchery,
                self.build_spine_crawler,
                self.build_spore_crawler,
                self.research_pneumatized_carapace,
                self.research_metabolic_boost,
                self.research_muscular_augments,
                self.research_glial_reconstitution,
                self.research_centrifugal_hooks,
                self.research_grooved_spines,
                self.research_adrenal_glands,
                self.research_chitinous_plating,
                self.research_anabolic_synthesis,
                self.research_melee_attack_3,
                self.research_missile_attack_3,
                self.research_ground_armor_3,
                self.research_flyer_attack_3,
                self.research_flyer_armor_3,
                self.scout_overlord,
                self.scout_zergling,
                self.move,
                self.attack,
                self.hunt,
                self.defend,
                self.wait]

    def build_net(self):
        n_actions = len(self.action_space)
        state_size = len(self.get_state())

        model = Sequential()
        model.add(Dense(200, activation='relu', input_shape=(state_size,)))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(n_actions))

        model.compile(optimizer=Adam(self.learning_rate), loss='mse')

        return model

    def get_state(self):
        if self.observation_space_mode == 'limited':
            # Resources
            state = [self.obs.observation.player.minerals,
                    self.obs.observation.player.vespene,
                    self.obs.observation.player.food_used,
                    self.obs.observation.player.food_cap - self.obs.observation.player.food_used]
    
            # Self units
                # Unit count
            state += [self.unit_count_agg(unit_name, with_burrowed=False) for unit_name in
                        ['Drone', 'Queen', 'Roach', 'Zergling', 'Hatchery']]
                # Requirements
            state += [(self.unit_count_agg(unit_name, with_burrowed=False, with_training=False) > 0) *  1
                    for unit_name in
                    ['RoachWarren', 'SpawningPool']]
                # Capabilities
            state += [(max([0] + [u.energy for u in self.get_units_agg('Queen', with_burrowed=False)])
                    >= 25) * 1] \
                + [any([u.order_id_0 == actions.RAW_FUNCTIONS.no_op.id
                        for u in self.get_units_agg(unit_name, with_burrowed=False)]) * 1
                    for unit_name in
                    ['Hatchery']]
    
            # Enemy units
            enemy_counts = [(len(self.memory.enemy_units[unit_type]), s) for unit_type, s in
                            [(units.Protoss.Immortal, 4),
                            (units.Protoss.Zealot, 2),
                            (units.Protoss.Stalker, 2),
                            (units.Protoss.Adept, 2),
                            (units.Protoss.VoidRay, 4),
                            (units.Protoss.Probe, 1)]]
            enemy_counts, enemy_unit_supplies = zip(*enemy_counts)
            state += list(enemy_counts)
    
            # Macro state
            state += [self.obs.observation.player.food_used - self.unit_count_agg('Drone')] \
                   + [sum([count * s for count, s in zip(enemy_counts, enemy_unit_supplies)])] \
                   + [sum([count * s for count, s in zip(enemy_counts, enemy_unit_supplies)])
                       - len(self.memory.enemy_units[units.Protoss.Probe])
                       - len(self.memory.enemy_units[units.Terran.SCV])
                       - len(self.memory.enemy_units[units.Zerg.Drone])]
    
            # Map features
            enemy_units_map = np.stack([self.obs.observation.feature_minimap.visibility_map
                                        == features.Visibility.VISIBLE,
                                        self.obs.observation.feature_minimap.player_relative
                                        == features.PlayerRelative.ENEMY]).all(axis=0).T.astype('uint8')
            if np.subtract(*self.self_coordinates) > 0:
                enemy_close_map = np.tril(enemy_units_map)
            else:
                enemy_close_map = np.triu(enemy_units_map)
            state += [min(enemy_close_map.sum(), 1)]
    
            return np.array(state)

        # Resources
        state = [self.obs.observation.player.minerals,
                 self.obs.observation.player.vespene,
                 self.obs.observation.player.food_used,
                 self.obs.observation.player.food_cap - self.obs.observation.player.food_used]

        # Self units
               # Unit count
        state += [self.unit_count_agg(unit_name, with_burrowed=False) for unit_name in
                    ['Baneling', 'BroodLord', 'Corruptor', 'Drone', 'Hydralisk', 'Larva',
                     'Mutalisk', 'Overlord', 'Overseer', 'Queen', 'Ravager', 'Roach', 'Ultralisk',
                     'Zergling', 'BanelingNest', 'CreepTumor', 'EvolutionChamber', 'Extractor',
                     'Hatchery', 'Lair', 'Hive', 'HydraliskDen', 'InfestationPit', 'RoachWarren',
                     'SpawningPool', 'SpineCrawler', 'Spire', 'GreaterSpire', 'SporeCrawler',
                     'UltraliskCavern', 'PneumatizedCarapace', 'MetabolicBoost',
                     'MuscularAugments', 'GlialReconstitution', 'CentrifugalHooks',
                     'GroovedSpines', 'AdrenalGlands', 'ChitinousPlating', 'AnabolicSynthesis']] \
               + [sum([self.unit_count_agg(lvl) for lvl in research_names]) for research_names in
                        [['MeleeAttack1', 'MeleeAttack2', 'MeleeAttack3'],
                         ['MissileAttack1', 'MissileAttack2', 'MissileAttack3'],
                         ['GroundArmor1', 'GroundArmor2', 'GroundArmor3'],
                         ['FlyerAttack1', 'FlyerAttack2', 'FlyerAttack3'],
                         ['FlyerArmor1', 'FlyerArmor2', 'FlyerArmor3']]]
               # Requirements
        state += [(self.unit_count_agg(unit_name, with_burrowed=False, with_training=False) > 0) *  1
                  for unit_name in
                  ['Corruptor', 'Roach', 'Zergling', 'BanelingNest', 'CreepTumor', 'Hatchery',
                   'Lair', 'Hive', 'HydraliskDen', 'InfestationPit', 'RoachWarren', 'SpawningPool',
                   'Spire', 'GreaterSpire', 'UltraliskCavern']]
               # Capabilities
        state += [(max([0] + [u.energy for u in self.get_units_agg('Queen', with_burrowed=False)])
                   >= 25) * 1] \
               + [any([u.order_id_0 == actions.RAW_FUNCTIONS.no_op.id
                       for u in self.get_units_agg(unit_name, with_burrowed=False)]) * 1
                  for unit_name in
                  ['EvolutionChamber', 'Hatchery', 'HydraliskDen', 'Spire', 'UltraliskCavern']]

        # Enemy units
        enemy_counts = [(len(self.memory.enemy_units[unit_type]), s) for unit_type, s in
                        [(units.Protoss.DarkTemplar, 2),
                         (units.Protoss.Archon, 4),
                         (units.Protoss.HighTemplar, 2),
                         (units.Protoss.Disruptor, 3),
                         (units.Protoss.Colossus, 6),
                         (units.Protoss.Immortal, 4),
                         (units.Protoss.Zealot, 2),
                         (units.Protoss.Stalker, 2),
                         (units.Protoss.Adept, 2),
                         (units.Protoss.PhotonCannon, 0),
                         (units.Protoss.Sentry, 2),
                         (units.Protoss.Probe, 1),
                         (units.Protoss.Carrier, 6),
                         (units.Protoss.Mothership, 8),
                         (units.Protoss.VoidRay, 4),
                         (units.Protoss.Tempest, 5),
                         (units.Protoss.WarpPrism, 2),
                         (units.Protoss.Oracle, 3),
                         (units.Protoss.Phoenix, 2),
                         (units.Protoss.Observer, 1),

                         (units.Terran.SiegeTankSieged, 3),
                         (units.Terran.Thor, 6),
                         (units.Terran.Ghost, 2),
                         (units.Terran.VikingAssault, 2),
                         (units.Terran.Hellbat, 2),
                         (units.Terran.Hellion, 2),
                         (units.Terran.Cyclone, 3),
                         (units.Terran.WidowMineBurrowed, 2),
                         (units.Terran.Marauder, 2),
                         (units.Terran.Reaper, 1),
                         (units.Terran.Marine, 1),
                         (units.Terran.Bunker, 0),
                         (units.Terran.SiegeTank, 3),
                         (units.Terran.MissileTurret, 0),
                         (units.Terran.SCV, 1),
                         (units.Terran.Battlecruiser, 6),
                         (units.Terran.Liberator, 3),
                         (units.Terran.Raven, 2),
                         (units.Terran.Banshee, 3),
                         (units.Terran.VikingFighter, 2),
                         (units.Terran.Medivac, 2),

                         (units.Zerg.Lurker, 3),
                         (units.Zerg.Ultralisk, 6),
                         (units.Zerg.Infestor, 2),
                         (units.Zerg.SwarmHost, 3),
                         (units.Zerg.Hydralisk, 2),
                         (units.Zerg.Ravager, 3),
                         (units.Zerg.Roach, 2),
                         (units.Zerg.Queen, 2),
                         (units.Zerg.SpineCrawler, 0),
                         (units.Zerg.SporeCrawler, 0),
                         (units.Zerg.Zergling, 0.5),
                         (units.Zerg.Drone, 1),
                         (units.Zerg.Baneling, 0.5),
                         (units.Zerg.BroodLord, 4),
                         (units.Zerg.Mutalisk, 2),
                         (units.Zerg.Corruptor, 2),
                         (units.Zerg.Viper, 3),
                         (units.Zerg.OverlordTransport, 0),
                         (units.Zerg.Overseer, 0)]]
        enemy_counts, enemy_unit_supplies = zip(*enemy_counts)
        state += list(enemy_counts)

        # Macro state
        state += [self.game_step] \
               + [sum([max(b.assigned_harvesters - b.ideal_harvesters, 0) for b in self.get_bases()])] \
               + [sum([max(b.ideal_harvesters - b.assigned_harvesters, 0) for b in self.get_bases()])] \
               + [sum([3 - u.assigned_harvesters for u in self.get_units_agg('Extractor')])] \
               + [any([u.buff_id_0 != buffs.Buffs.QueenSpawnLarvaTimer
                       for u in self.get_units_agg('Hatchery')]) * 1] \
               + [self.obs.observation.player.food_used - self.unit_count_agg('Drone')] \
               + [sum([count * s for count, s in zip(enemy_counts, enemy_unit_supplies)])] \
               + [sum([count * s for count, s in zip(enemy_counts, enemy_unit_supplies)])
                  - len(self.memory.enemy_units[units.Protoss.Probe])
                  - len(self.memory.enemy_units[units.Terran.SCV])
                  - len(self.memory.enemy_units[units.Zerg.Drone])]

        # Map features
        enemy_units_map = np.stack([self.obs.observation.feature_minimap.visibility_map
                                    == features.Visibility.VISIBLE,
                                    self.obs.observation.feature_minimap.player_relative
                                    == features.PlayerRelative.ENEMY]).all(axis=0).T.astype('uint8')
        if np.subtract(*self.self_coordinates) > 0:
            enemy_close_map = np.tril(enemy_units_map)
        else:
            enemy_close_map = np.triu(enemy_units_map)
        state += [np.array(self.obs.observation.feature_minimap.creep.T).sum()
                  / self.obs.observation.feature_minimap.creep.size] \
               + [min(enemy_close_map.sum(), 1)]

        return np.array(state)

    def learn(self, s, q, a, r, q_, is_terminal=False):
        target = q
        target[a] = r + self.discount * max(q_) * (not is_terminal)

        if len(self.replay_memory) < self.memory_size:
            self.replay_memory.append(None)
        self.replay_memory[self.position] = (s, target)
        self.position = (self.position + 1) % self.memory_size

        if len(self.replay_memory) >= self.batch_size:
            batch = rd.sample(self.replay_memory, self.batch_size)
            X, y = zip(*batch)
            self.net.train_on_batch(np.array(X), np.array(y))

    def step(self, obs):
        super().step(obs)

        if self.net is None:
            self.net = self.build_net()

        self.cast_inject_larva_hl()
        self.adjust_workers_distribution_intra_hl(timeout=150)
        self.adjust_workers_distribution_inter_hl(timeout=150)
        pop = self.obs.observation.player.food_used
        cap = self.obs.observation.player.food_cap
        if cap >= 66 and pop >= cap - 10 and cap < 200:
            self.train_overlord_hl(self.unit_count_agg('Overlord', with_training=False) + 2)
        if self.obs.observation.player.food_used >= 190:
            ZerglingRush.attack_then_hunt(self)

        if self.action is not None:
            return self.action

        state = self.get_state()
        q_values = self.net.predict(state.reshape((1, -1))).ravel()
        if np.random.random() > self.epsilon:
            action = np.argmax(q_values)
        else:
            action = np.random.randint(len(self.action_space))
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        if self.previous_action is not None:
            self.learn(self.previous_state,
                       self.previous_q_values,
                       self.previous_action,
                       obs.reward,
                       q_values,
                       is_terminal=self.obs.last()
                                   or self.game_step * STEP_MUL > self.episode_length)

        self.previous_state = state
        self.previous_q_values = q_values
        self.previous_action = action

        self.action = self.action_space[action]()

        self.wait_hl()

        self.rewards[-1].append(self.obs.reward)

        return self.action
