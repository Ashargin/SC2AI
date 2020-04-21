import numpy as np
from pysc2.lib import actions
from raw_base_agent import ZergAgent


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
    def __init__(self):
        super().__init__()
        self.finished_opening = False
        self.finished_followup = False
        self.finished_midgame = False
        self.midgame_increase = None

    def reset(self):
        super().reset()
        self.finished_opening = False
        self.finished_followup = False
        self.finished_midgame = False
        self.midgame_increase = None
        ##################
        self.attack_ended = False

    def essentials(self):
        self.cast_inject_larva_hl()
        self.move_workers_to_gas_hl(timeout=150)
        self.adjust_workers_distribution_intra_hl(timeout=150)
        self.adjust_workers_distribution_inter_hl(timeout=150)
        self.set_rally_point_units_hl(coordinates='home')
        self.scout_overlord_hl(coordinates='enemy_close', timeout=3000)
        if self.game_step % 10 == 0:
            self.defend_hl(defend_up_to=0.6, with_queen=False)
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

    def midgame_roach_hydra(self):
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
        self.build_extractor_hl(8, how=5, pop=165, increase=increase)

        if self.obs.observation.player.food_used >= 165 and self.action is None:
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

        self.build_ultralisk_cavern_hl()
        self.research_chitinous_plating_hl()
        self.research_ground_armor_2_hl()
        self.build_spire_hl()
        self.build_greater_spire_hl()
        self.research_flyer_attack_1_hl()
        self.research_melee_attack_1_hl()
        self.build_spire_hl(2)
        self.research_flyer_armor_1_hl()
        self.research_flyer_attack_2_hl()
        self.research_adrenal_glands_hl()
        self.research_melee_attack_2_hl()
        self.research_flyer_armor_2_hl()
        self.research_flyer_attack_3_hl()
        self.research_flyer_armor_3_hl()
        self.research_melee_attack_3_hl()
        self.wait_hl(pop=200, increase=increase)

    def step(self, obs):
        super().step(obs)

        # Build order
        # Always needed
        self.essentials()
        self.opening()
        self.followup_standard()
        self.midgame_baneling()
        self.lategame_ultralisk_broodlord()

        ######################################
        if self.game_step % 100 == 0:
            print(self.game_step)
        if self.obs.observation.player.food_used >= 195:
            if self.game_step % 2 == 0 or self.unit_count_agg('Corruptor', with_training=False) == 0:
                ZerglingRush.attack_then_hunt(self, can_reach_air=False,
                                              unit_names=['Zergling', 'Baneling',
                                                          'Roach', 'Hydralisk',
                                                          'Ultralisk', 'BroodLord'])
            else:
                ZerglingRush.attack_then_hunt(self, can_reach_ground=False,
                                              unit_names=['Corruptor'])

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
    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()

    def essentials(self):
        self.cast_inject_larva_hl()
        self.move_workers_to_gas_hl(timeout=150)
        self.adjust_workers_distribution_intra_hl(timeout=150)
        self.adjust_workers_distribution_inter_hl(timeout=150)
        self.set_rally_point_units_hl(coordinates='home')
        self.scout_overlord_hl(coordinates='enemy_close', timeout=3000)
        if self.game_step % 10 == 0:
            self.defend_hl(defend_up_to=0.6, with_queen=False)
        self.build_creep_tumor_queen_hl(10)
        self.build_creep_tumor_tumor_hl()

        pop = self.obs.observation.player.food_used
        cap = self.obs.observation.player.food_cap
        if cap >= 66 and pop >= cap - 10 and cap < 200:
            self.train_overlord_hl(self.unit_count_agg('Overlord', with_training=False) + 2)

    def step(self, obs):
        super().step(obs)

        self.essentials()

        if self.action is None:
            pass

        self.wait_hl()
        return self.action
