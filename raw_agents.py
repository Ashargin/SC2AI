import numpy as np
from raw_base_agent import ZergAgent


class ZerglingRush(ZergAgent):
    """Build : https://lotv.spawningtool.com/build/119713/"""

    def __init__(self):
        super().__init__()
        self.mode = 'build'
        self.zerglings_to_push = 12

    def reset(self):
        super().reset()
        self.mode = 'build'

    def update_mode(self):
        
        zerglings = [u for u in self.get_units_agg('Zergling') \
                       if np.abs(np.array(self.enemy_coordinates) - (u.x, u.y)).sum()
                       <= np.abs(np.array(self.self_coordinates) - (u.x, u.y)).sum()]

        if self.mode == 'build' and len(zerglings) >= 2 + self.zerglings_to_push:
            self.mode = 'attack'
        if self.mode == 'attack' and len(zerglings) <= int((2 + self.zerglings_to_push) / 2):
            self.mode = 'build'

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
            self.attack_hl(can_reach_air=False, where='head')

        self.wait_hl()

        return self.action


class MacroZerg(ZergAgent):
    """Build : https://lotv.spawningtool.com/build/118526/"""

    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()

    def step(self, obs):
        super().step(obs)

        # Build order
        # Always needed
        self.scout_overlord_hl(coordinates='enemy_close', timeout=np.inf)
        self.cast_inject_larva_hl()
        self.build_creep_tumor_queen_hl()
        self.build_creep_tumor_tumor_hl()
        self.set_rally_point_units_hl(coordinates='home')
        self.move_workers_to_gas_hl()
        self.adjust_workers_distribution_inter_hl()

        # Building mode
        ### missing lings?
        ### add drone count
        self.train_overlord_hl(2, pop=13)
        self.build_hatchery_hl(2, pop=16)
        self.build_spawning_pool_hl(pop=19)
        self.build_extractor_hl(1, pop=20)
        self.train_overlord_hl(3, pop=25)
        self.train_queen_hl(1, how=1, pop=25)
        self.train_overlord_hl(5, pop=28)
        self.train_queen_hl(2, how=2, pop=28)
        self.build_hatchery_hl(3, pop=28)
        self.build_roach_warren_hl(pop=33)
        self.train_overlord_hl(6, pop=33)
        self.build_lair_hl(pop=33)
        self.build_evolution_chamber_hl(1, pop=33)
        self.train_queen_hl(3, how=3, pop=34)
        self.research_missile_attack_1_hl(pop=42)
        self.build_extractor_hl(3, pop=47)
        self.train_queen_hl(4, pop=47)
        self.research_glial_reconstitution_hl(pop=47)
        self.train_roach_hl(4, pop=51)
        self.train_overseer_hl(1, pop=59)
        self.train_roach_hl(8, pop=59)
        self.build_hatchery_hl(4, pop=82)
        self.build_extractor_hl(5, pop=81)
        self.build_hydralisk_den_hl(pop=90)
        self.build_evolution_chamber_hl(2, pop=89)
        self.build_extractor_hl(6, pop=89)
        self.train_overseer_hl(2, pop=101)
        self.build_hatchery_hl(5, pop=104)
        self.build_extractor_hl(8, pop=103)
        self.train_hydralisk_hl(5, pop=103)
        self.research_grooved_spines_hl(pop=103)
        self.build_infestation_pit_hl(pop=116)
        self.research_missile_attack_2_hl(pop=116)
        self.research_ground_armor_1_hl(pop=116)
        self.train_hydralisk_hl(10, pop=116)
        self.build_hive_hl(pop=150)
        self.research_muscular_augments_hl(pop=172)
        self.train_roach_hl(10, pop=190)
        self.build_lurker_den_hl(pop=200)
        self.build_spire_hl(pop=200)

        self.wait_hl()

        return self.action
