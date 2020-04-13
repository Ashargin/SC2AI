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


class HydraliskPush(ZergAgent):
    """Build : https://lotv.spawningtool.com/build/118355/"""

    def __init__(self):
        super().__init__()
        self.mode = 'macro'
        self.hydralisks_to_push = 10

    def reset(self):
        super().reset()
        self.mode = 'macro'

    def step(self, obs):
        super().step(obs)

        n_hydralisks = self.unit_count_agg('Hydralisk', with_training=False)

        # Build order
        # Always needed
        self.cast_inject_larva_hl()
        self.adjust_workers_distribution_inter_hl()

        if self.mode == 'macro':
            # First base
            self.train_drone_hl(14, how=1)
    
            # Second base
            self.build_hatchery_hl(2, pop=14)
            self.train_drone_hl(28, how=2)
            self.build_spine_crawler_hl(2, how=2)
    
            # Third base
            self.train_queen_hl(1)
            self.build_hatchery_hl(3, pop=28)
            self.build_spine_crawler_hl(4, how=2)
    
            # Train hydralisks and attack
            self.train_hydralisk_hl(5)
            self.train_drone_hl(38, how=2)
            self.train_hydralisk_hl(100)
            if self.game_step % 10 == 0:
                # Keep army at second base
                self.move_hl(coordinates='home')
    
            # Research in the meantime
            self.research_missile_attack_1_hl()
            self.research_ground_armor_1_hl()
            self.research_muscular_augments_hl()
            self.research_ground_armor_2_hl()
            self.research_grooved_spines_hl()
            self.research_missile_attack_2_hl()

            if n_hydralisks >= self.hydralisks_to_push:
                self.mode = 'attack'

        elif self.mode == 'attack':
            self.attack_hl(where='head')

            if n_hydralisks <= int(self.hydralisks_to_push / 2):
                self.mode = 'macro'

        self.wait_hl()

        return self.action