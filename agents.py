import os
import numpy as np
import random as rd
from pysc2.agents import base_agent
from pysc2.lib import actions, buffs, features, units, upgrades
from memory import Memory
from settings import RESOLUTION

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS


class BaseAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.log = None # debug tool
        self.time_step = 0
        self.memory = Memory()
        self.action = None

    def reset(self):
        super().reset()
        if self.log is not None:
            self.log.close()
        self.log = open(f'{os.path.abspath(os.getcwd())}'
                        f'\\data\\logs\\log_{self.__class__.__name__}_game_{self.episodes-1}.txt', 'w')
        self.time_step = 0
        self.memory = Memory()

    def step(self, obs):
        super().step(obs)
        self.obs = obs
        self.action = None
        self.memory.update(self.obs)

        if obs.first():
            self.log.write('### Game log ###')

        # Log data
        self.log.write(f'\n\nTime step {self.time_step}')
        self.log.write(f'\nResources : {self.obs.observation.player.minerals}, '
                       f'{self.obs.observation.player.vespene}')
        self.log.write(f'\nSupply : {self.obs.observation.player.food_used}'
                       f'/{self.obs.observation.player.food_cap}')
        self.log.write(f'\nSelected : {self.obs.observation.single_select}, '
                       f'{self.obs.observation.multi_select}')
        self.log.write(f'\nLarvae : {self.unit_count(units.Zerg.Larva)}')

        self.time_step += 1

    # Core methods
    def unit_count(self, unit_type):
        return len(self.get_units_by_type(unit_type))

    def get_units_by_type(self, unit_type):
        return [unit for unit in self.obs.observation.feature_units
                     if unit.unit_type == unit_type]

    def unit_type_is_selected(self, unit_type):
        if unit_type in [u.unit_type for u in self.obs.observation.single_select]:
            return True
        if unit_type in [u.unit_type for u in self.obs.observation.multi_select]:
            return True
        return False

    def available(self, action):
        return action.id in self.obs.observation.available_actions

    def do_if_available(self, action, *args, raise_error=True):
        self.log.write(f'\nAvailable actions : {self.obs.observation.available_actions}')
        if self.available(action):
            self.log.write(f'\nChose action {action}, args {list(args)}')
            return action(*args)
        if raise_error:
            raise Warning(f'Cannot perform action {action}')
        return FUNCTIONS.no_op()

    # Base actions
    def try_select(self, unit_type, how='naive', completed=True):
        units = self.get_units_by_type(unit_type)
        if completed:
            units = [u for u in units if u.build_progress == 100]
        if len(units) > 0:
            if how == 'naive':
                picked = rd.choice(units)
                x, y = picked.x, picked.y
            else:
                x, y = how
            x = min(max(x, 0), RESOLUTION - 1)
            y = min(max(y, 0), RESOLUTION - 1)
            return self.do_if_available(FUNCTIONS.select_point,
                                        'select', (x, y))
        raise Warning(f'No available {unit_type} to be selected')

    def try_build(self, action, how='naive'):
        if self.unit_type_is_selected(self.worker_type) \
                and all([u[2] > 0 for u in self.obs.observation.single_select]) \
                and all([u[2] > 0 for u in self.obs.observation.multi_select]):
            if how == 'naive':
                x = np.random.randint(0, RESOLUTION - 1)
                y = np.random.randint(0, RESOLUTION - 1)
            else:
                x, y = how
            x = min(max(x, 0), RESOLUTION - 1)
            y = min(max(y, 0), RESOLUTION - 1)
            return self.do_if_available(action,
                                        'now', (x, y), raise_error=False)
        return self.try_select(self.worker_type)

    def try_train(self, action, trainer, how='naive'):
        if self.unit_type_is_selected(trainer) \
                and all([u[2] > 0 for u in self.obs.observation.single_select]) \
                and all([u[2] > 0 for u in self.obs.observation.multi_select]):
            return self.do_if_available(action, 'now')
        return self.try_select(trainer, how=how)

    def try_attack(self, coordinates, how='naive'):
        if 7 in self.obs.observation.last_actions:
            return self.do_if_available(FUNCTIONS.Attack_minimap, 'now', coordinates)
        else:
            return self.do_if_available(FUNCTIONS.select_army, 'select')


class ZergAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.race_name = 'zerg'
        self.worker_type = units.Zerg.Drone

    # Core methods
    def get_hatcheries(self):
        return self.get_units_by_type(units.Zerg.Hive) \
             + self.get_units_by_type(units.Zerg.Lair) \
             + self.get_units_by_type(units.Zerg.Hatchery)

    def unit_count_with_training(self, unit_type, trainer=units.Zerg.Cocoon, mul=1):
        if unit_type == units.Zerg.Zergling:
            mul = 2
        live_count = self.unit_count(unit_type)
        raw_train_id = getattr(RAW_FUNCTIONS, f'Train_{unit_type.name}_quick').id
        training_count = sum([t.order_id_0 == raw_train_id
                                for t in self.get_units_by_type(trainer)])
        return live_count + training_count * mul

    # Base actions
    def try_train(self, action, trainer=units.Zerg.Larva, how='naive'):
        if trainer == units.Zerg.Larva and self.unit_count(units.Zerg.Larva) == 0:
            return FUNCTIONS.no_op()
        return super().try_train(action, trainer, how=how)

    # Specific actions
    def wait_train_drone(self):
        if self.obs.observation.player.minerals >= 50 \
                    and self.obs.observation.player.food_cap \
                    >= self.obs.observation.player.food_used + 1:
            return self.try_train(FUNCTIONS.Train_Drone_quick)
        else:
            return FUNCTIONS.no_op()

    def try_build_extractor(self):
        neutral_vespenes = self.get_units_by_type(units.Neutral.VespeneGeyser)
        hatcheries = self.get_hatcheries()
        hatchery = hatcheries[int(self.unit_count(units.Zerg.Extractor) / 2)]
        dists = [(v.x - hatchery.x)**2 + (v.y - hatchery.y)**2 for v in neutral_vespenes]
        vespene = neutral_vespenes[np.argmin(dists)]
        return self.try_build(FUNCTIONS.Build_Extractor_screen, how=(vespene.x, vespene.y))

    def try_move_worker_to_gas(self):
        extractors = self.get_units_by_type(units.Zerg.Extractor)
        extractors = [e for e in extractors if e.build_progress == 100]
        extractor = [e for e in extractors if e.assigned_harvesters < 3][0]
        workers = self.get_units_by_type(self.worker_type)
        workers = [w for w in workers if w.buff_id_0 in (271, 272) and w.order_id_0 == 360]
        if not workers:
            return FUNCTIONS.no_op()

        dists = [(w.x - extractor.x)**2 + (w.y - extractor.y)**2 for w in workers]
        if sum([w.is_selected for w in workers]) == 1 \
                and all([u[2] > 0 for u in self.obs.observation.single_select]) \
                and len(self.obs.observation.single_select) == 1:
            selected = [w for w in workers if w.is_selected == 1][0]
            dists.sort()
            if (selected.x - extractor.x)**2 + (selected.y - extractor.y)**2 \
                                                            <= dists[:3][-1]:
                return self.do_if_available(FUNCTIONS.Harvest_Gather_screen,
                                            'now', (extractor.x, extractor.y))

        w = workers[np.argmin(dists)]
        return self.try_select(self.worker_type, how=(w.x, w.y))

    def try_move_worker_to_minerals(self):
        workers = self.get_units_by_type(self.worker_type)
        workers = [w for w in workers if w.buff_id_0 == 275 and w.order_id_0 == 360]
        if not workers:
            return FUNCTIONS.no_op()

        if sum([w.is_selected for w in workers]) == 1 \
                and all([u[2] > 0 for u in self.obs.observation.single_select]) \
                and len(self.obs.observation.single_select) == 1:
            selected = [w for w in workers if w.is_selected == 1][0]
            minerals = self.get_units_by_type(units.Neutral.MineralField)
            dists = [(m.x - selected.x)**2 + (m.y - selected.y)**2 for m in minerals]
            m = minerals[np.argmin(dists)]
            return self.do_if_available(FUNCTIONS.Harvest_Gather_screen,
                                        'now', (m.x, m.y))

        extractors = self.get_units_by_type(units.Zerg.Extractor)
        extractor = [e for e in extractors if e.assigned_harvesters > 0][0]
        dists = [(w.x - extractor.x)**2 + (w.y - extractor.y)**2 for w in workers]
        w = workers[np.argmin(dists)]
        return self.try_select(self.worker_type, how=(w.x, w.y))


class ZerglingRush(ZergAgent):
    """Build : https://lotv.spawningtool.com/build/118355/"""

    def __init__(self):
        super().__init__()
        self.attack_coordinates = None

    def step(self, obs):
        super().step(obs)
        pop = self.obs.observation.player.food_used
        cap = self.obs.observation.player.food_cap
        minerals = self.obs.observation.player.minerals
        vespene = self.obs.observation.player.vespene
        player_relative = self.obs.observation.feature_minimap.player_relative
        raw_units = self.obs.observation.raw_units
        n_workers = self.unit_count_with_training(self.worker_type)

        # Get attack coordinates
        if obs.first():
            hatchery = [u for u in raw_units if u.unit_type == 86][0]
            self.attack_coordinates = (RESOLUTION - 1 - hatchery.x, RESOLUTION - 1 - hatchery.y)

        # Build order
        # 14 Hatchery
        if not self.unit_count(units.Zerg.Hatchery) >= 2:
            if n_workers >= 14:
                if minerals >= 300:
                    return self.try_build(FUNCTIONS.Build_Hatchery_screen)
                else:
                    return FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 14 Extractor
        if not self.unit_count(units.Zerg.Extractor) >= 1:
            if n_workers >= 14:
                if minerals >= 25:
                    return self.try_build_extractor()
                else:
                    return FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 14 Spawning pool
        if not self.unit_count(units.Zerg.SpawningPool) >= 1:
            if n_workers >= 14:
                if minerals >= 200:
                    return self.try_build(FUNCTIONS.Build_SpawningPool_screen)
                else:
                    return FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 14 Overlord
        if not self.unit_count_with_training(units.Zerg.Overlord) >= 2:
            if n_workers >= 14:
                if minerals >= 100:
                    return self.try_train(FUNCTIONS.Train_Overlord_quick)
                else:
                    return FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # Move drones to gas until 100 vespene are harvested
        spawning_pools = self.get_units_by_type(units.Zerg.SpawningPool)
        metabolic_boost_researched = 495 in [p.order_id_0 for p in spawning_pools] \
                                    or 66 in self.obs.observation.upgrades
        extractors = self.get_units_by_type(units.Zerg.Extractor)
        if not (sum([e.assigned_harvesters for e in extractors]) >= 3 \
                    or vespene >= 100 \
                    or metabolic_boost_researched):
            if any([e.build_progress == 100 for e in extractors]):
                return self.try_move_worker_to_gas()
            else:
                return FUNCTIONS.no_op()

        # Move drones back to minerals once 100 vespene are harvested
        if not (sum([e.assigned_harvesters for e in extractors]) <= 0 \
                    or (vespene < 100 and not metabolic_boost_researched)):
            return self.try_move_worker_to_minerals()

        # 14 Research metabolic boost
        if not metabolic_boost_researched:
            if n_workers >= 14:
                if minerals >= 100 and vespene >= 100:
                    return self.try_train(FUNCTIONS.Research_ZerglingMetabolicBoost_quick,
                                        trainer=units.Zerg.SpawningPool)
                else:
                    return FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 16 Queen
        if not self.unit_count_with_training(units.Zerg.Queen, trainer=units.Zerg.Hatchery) >= 1:
            if n_workers >= 16:
                if minerals >= 150:
                    return self.try_train(FUNCTIONS.Train_Queen_quick,
                                        trainer=units.Zerg.Hatchery)
                else:
                    return FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 18 Zergling x5 ==> 23
        if not pop >= 23:
            if n_workers >= 16:
                if minerals >= 50:
                    return self.try_train(FUNCTIONS.Train_Zergling_quick)
                else:
                    return FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 23 Overlord
        if not self.unit_count_with_training(units.Zerg.Overlord) >= 3:
            if n_workers >= 16:
                if minerals >= 100:
                    return self.try_train(FUNCTIONS.Train_Overlord_quick)
                else:
                    return FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 23 Zergling x11 ==> 34
        if not pop >= 34:
            if n_workers >= 16:
                if minerals >= 50:
                    return self.try_train(FUNCTIONS.Train_Zergling_quick)
                else:
                    return FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 34 attack
        return self.try_attack(self.attack_coordinates)

        ## Todo
        # Use all larvae (second base) (out of screen)
        # Keep creating lings
        # Keep creating overlords
        # Inject larvae
        # Rally point second base on minerals (idle workers)
        # Send zergling close to enemy base before 34
        # Set rally point of bases when attack is launched
        # Second queen?
        # Micro

        # Compte units dict
        # Plus haut niveau
        # no hard coded
