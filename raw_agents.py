import numpy as np
from raw_base_agent import ZergAgent


class ZerglingRush(ZergAgent):
    """Build : https://lotv.spawningtool.com/build/119713/"""

    def __init__(self):
        super().__init__()
        self.mode = 'build'
        self.zerglings_to_push = 14

    def reset(self):
        super().reset()
        self.mode = 'build'

    def step(self, obs):
        super().step(obs)

        zerglings = [u for u in self.get_units_agg('Zergling') \
                       if np.abs(np.array(self.enemy_coordinates) - (u.x, u.y)).sum()
                       <= np.abs(np.array(self.memory.base_locations[0]) - (u.x, u.y)).sum()]
        

        # Build order
        if self.mode == 'build':
            self.build_spawning_pool_hl()
            self.train_zergling_hl(6, pop=14)
            self.build_hatchery_hl(2)
            self.train_zergling_hl(10)
            # self.train_queen_hl(1) needs to inject larva
            self.build_spine_crawler_hl(1, pop=24, increase=self.train_zergling, how=2)
            self.train_zergling_hl(400)
            self.move_hl(0.4 * np.array(self.memory.base_locations[0])
                       + 0.6 * np.array(self.enemy_coordinates))

            if len(zerglings) >= self.zerglings_to_push:
                self.mode = 'attack'

        elif self.mode == 'attack':
            self.train_zergling_hl(400)
            self.attack_hl(can_reach_air=False, where='head')

            if len(zerglings) <= int(self.zerglings_to_push / 2):
                self.mode = 'build'

        self.wait_hl()

        return self.action


class HydraliskPush(ZergAgent):
    """Build : https://lotv.spawningtool.com/build/118355/"""

    def __init__(self):
        super().__init__()
        self.mode = 'macro'
        self.hydralisks_to_push = 30

    def reset(self):
        super().reset()
        self.mode = 'macro'

    def step(self, obs):
        super().step(obs)

        n_hydralisks = self.unit_count_agg('Hydralisk', with_training=False)

        # Build order
        if self.mode == 'macro':
            # First base
            self.train_drone_hl(14, how=1)
    
            # Second base
            self.build_hatchery_hl(2, pop=14)
            self.train_drone_hl(28, how=2)
            self.move_workers_to_minerals_hl(how=2, only_idle=True)
            self.build_spine_crawler_hl(2, how=2)
    
            # Third base
            self.build_hatchery_hl(3, pop=28)
            self.build_spine_crawler_hl(4, how=2)
    
            # Train hydralisks and attack
            self.train_hydralisk_hl(5)
            self.train_drone_hl(38, how=2)
            self.train_hydralisk_hl(100)
            if self.game_step % 10 == 0:
                # Keep army at second base
                self.move_hl(coordinates=0.99 * np.array(self.memory.base_locations[:2][-1])
                                       + 0.01 * np.array(self.enemy_coordinates))
    
            # Research in the meantime
            self.build_evolution_chamber_hl(2, pop=28)
            self.research_missile_attack_1_hl()
            self.research_ground_armor_1_hl()
            self.research_muscular_augments_hl()
            self.research_ground_armor_2_hl()
            self.research_grooved_spines_hl()
            self.research_missile_attack_2_hl()

            if n_hydralisks >= self.hydralisks_to_push:
                self.mode = 'attack'

        elif self.mode == 'attack':
            self.attack_hl(only_idle=True)

            if n_hydralisks <= int(self.hydralisks_to_push / 2):
                self.mode = 'macro'

        self.wait_hl()

        return self.action