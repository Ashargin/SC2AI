import os
import numpy as np
import random as rd
import cv2
import scipy
from pysc2.agents import base_agent
from pysc2.lib import actions, buffs, features, units, upgrades
from memory import Memory
from settings import RESOLUTION, THRESHOLD

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS


class RawAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.raw_interface = True
        self.log = None # debug tool
        self.game_step = 0
        self.action = None
        self.memory = Memory()

    def reset(self):
        super().reset()
        if self.log is not None:
            self.log.close()
        self.log = open(f'{os.path.abspath(os.getcwd())}'
                        f'\\data\\logs\\log_{self.__class__.__name__}_game_{self.episodes-1}.txt', 'w')
        self.game_step = 0
        self.memory = Memory()

    def step(self, obs):
        super().step(obs)
        self.obs = obs
        self.action = None
        self.memory.update(self.obs)

        if obs.first():
            self.log.write('### Game log ###')

        # Log data
        self.log.write(f'\n\nTime step {self.game_step}')
        self.log.write(f'\nResources : {self.obs.observation.player.minerals}, '
                       f'{self.obs.observation.player.vespene}')
        self.log.write(f'\nSupply : {self.obs.observation.player.food_used}'
                       f'/{self.obs.observation.player.food_cap}')
        self.log.write(f'\nSelected : {self.obs.observation.single_select}, '
                       f'{self.obs.observation.multi_select}')
        self.log.write(f'\nLarvae : {self.unit_count(units.Zerg.Larva)}')

        self.game_step += 1

    # Core methods
    def unit_count(self, unit_type):
        return len(self.get_units_by_type(unit_type))

    def get_units_by_type(self, unit_type):
        if unit_type in self.memory.self_units:
            return self.memory.self_units[unit_type]
        if unit_type in self.memory.neutral_units:
            return self.memory.neutral_units[unit_type]
        return []

    def do(self, action, *args, raise_error=True):
        self.log.write(f'\nChose action {action}, args {list(args)}')
        return action(*args)

    # Base actions
    def build(self, action, builder=None, how=1, where='random'):
        if builder is None:
            builder = self.worker_type

        # Choose builder
        builders = self.get_units_by_type(builder)
        if how == 'base':
            x, y = self.memory.base_locations[len(self.get_bases()) - 1]
            dists = [(b.x - x)**2 + (b.y - y)**2 for b in builders]
            picked = builders[np.argmin(dists)]
            tag = picked.tag
        elif how < 50: # how is the number of the base to train from
            x, y = self.memory.base_locations[how - 1]
            dists = [(b.x - x)**2 + (b.y - y)**2 for b in builders]
            picked = builders[np.argmin(dists)]
            tag = picked.tag
        else: # how is the tag of the builder
            tag = how

        # Choose target location / tag
        if where == 'random': # tag target cannot be random, must be deterministic
            # Create the matrix of possible locations
            creep_constraint = np.ones((RESOLUTION, RESOLUTION))
            if self.race_name == 'zerg' and action != RAW_FUNCTIONS.Build_Hatchery_pt:
                creep_constraint = self.obs.observation.feature_minimap.creep
            can_build = np.stack([self.obs.observation.feature_minimap.pathable,
                                  self.obs.observation.feature_minimap.buildable,
                                  creep_constraint]).all(axis=0).T.astype('uint8')
            diam = int(RESOLUTION / 36) # approximate standard building size
            for u in self.obs.observation.raw_units:
                #######################################################
                # IF ON LAND ??? (air, ?burrowed? etc)
                can_build[max(u.x - diam, 0):min(u.x + diam, RESOLUTION-1),
                          max(u.y - diam, 0):min(u.y + diam, RESOLUTION-1)] = 0
            if action in [RAW_FUNCTIONS.Build_CommandCenter_pt,
                          RAW_FUNCTIONS.Build_Nexus_pt,
                          RAW_FUNCTIONS.Build_Hatchery_pt]:
                diam = int(RESOLUTION / 27) # approximate base building size
            can_build = cv2.erode(can_build, np.ones((diam, diam)))

            # Choose the marker point
            if how == 'base':
                minerals = self.obs.observation.feature_minimap.player_relative.T == 3
                # Minerals proximity score :
                n = int(THRESHOLD * 1.5)
                n += 1 - n % 2
                kernel = np.sqrt([[(i - (n-1)/2)**2 + (j - (n-1)/2)**2 for j in range(n)] for i in range(n)])
                kernel = kernel.max() - kernel
                best_spots = scipy.signal.convolve2d(minerals, kernel, mode='same')

                minerals = np.argwhere(minerals)
                bases = self.get_bases()
                dists = np.array([[(c[0] - b.x)**2 + (c[1] - b.y)**2 for b in bases] for c in minerals])
                dists = np.sqrt(dists).min(axis=1)
                minerals, dists = minerals[dists > THRESHOLD], dists[dists > THRESHOLD]
                x, y = minerals[np.argmin(dists)]
            else:
                base = how if how < 50 else 1
                x, y = self.memory.base_locations[base - 1]
                best_spots = None # use closest

            # Find the best location near the marker point
            n = int(THRESHOLD * 0.75)
            can_build[:max(x-n, 0), :] = can_build[min(x+n, RESOLUTION-1):, :] \
                = can_build[:, :max(y-n, 0)] = can_build[:, min(y+n, RESOLUTION-1):] = 0
            candidates = np.argwhere(can_build == 1)
            if best_spots is None:
                scores = [-(c[0] - x)**2 - (c[1] - y)**2 for c in candidates]
            else:
                scores = best_spots[candidates[:, 0], candidates[:, 1]]
            target = tuple(candidates[np.argmax(scores)])
        else: # where is the location, or tag, of the target unit
            target = where

        return self.do(action, 'now', tag, target)

    def train(self, action, trainer, how='random'):
        trainers = self.get_units_by_type(trainer)
        if how == 'random':
            picked = rd.choice(trainers)
        else: # how is the number of the base to train from
            x, y = self.memory.base_locations[how - 1]
            dists = [(t.x - x)**2 + (t.y - y)**2 for t in trainers]
            picked = trainers[np.argmin(dists)]

        return self.do(action, 'now', picked.tag)

    def attack(self, attackers, coordinates):
        attackers = [u.tag for u in attackers]
        return self.do(RAW_FUNCTIONS.Attack_minimap, 'now', attackers, coordinates)


class ZergAgent(RawAgent):
    def __init__(self):
        super().__init__()
        self.race_name = 'zerg'
        self.worker_type = units.Zerg.Drone

    # Core methods
    def get_bases(self):
        return self.get_units_by_type(units.Zerg.Hatchery) \
             + self.get_units_by_type(units.Zerg.Lair) \
             + self.get_units_by_type(units.Zerg.Hive)

    def unit_count_with_training(self, unit_type, mul=1):
        if unit_type == units.Zerg.Zergling:
            mul = 2
        if unit_type == units.Zerg.Queen:
            trainers = self.get_bases()
        else:
            trainers = self.get_units_by_type(units.Zerg.Cocoon)
        live_count = self.unit_count(unit_type)
        raw_train_id = getattr(RAW_FUNCTIONS, f'Train_{unit_type.name}_quick').id
        training_count = sum([t.order_id_0 == raw_train_id for t in trainers])
        return live_count + training_count * mul

    # Base actions
    def train(self, action, trainer=units.Zerg.Larva, how='random'):
        if trainer == units.Zerg.Larva and self.unit_count(units.Zerg.Larva) == 0:
            return RAW_FUNCTIONS.no_op()
        return super().train(action, trainer, how=how)

    # Specific actions
    def wait_train_drone(self, how='random'):
        if self.obs.observation.player.minerals >= 50 \
                    and self.obs.observation.player.food_cap \
                    >= self.obs.observation.player.food_used + 1:
            return self.train(RAW_FUNCTIONS.Train_Drone_quick, how=how)
        else:
            return RAW_FUNCTIONS.no_op()

    def build_hatchery(self):
        return self.build(RAW_FUNCTIONS.Build_Hatchery_pt, how='base')

    def build_extractor(self, how=1):
        neutral_vespenes = self.get_units_by_type(units.Neutral.VespeneGeyser)
        x, y = self.memory.base_locations[how - 1]
        dists = [(v.x - x)**2 + (v.y - y)**2 for v in neutral_vespenes]
        vespene = neutral_vespenes[np.argmin(dists)]
        return self.build(RAW_FUNCTIONS.Build_Extractor_unit, how=how, where=vespene.tag)

    def move_worker_to_gas(self):
        extractors = self.get_units_by_type(units.Zerg.Extractor)
        extractors = [e for e in extractors if e.build_progress == 100]
        extractor = [e for e in extractors if e.assigned_harvesters < 3][0]
        workers = self.get_units_by_type(self.worker_type)
        workers = [w for w in workers if w.buff_id_0 in (271, 272) and w.order_id_0 == 360]
        if not workers:
            return RAW_FUNCTIONS.no_op()

        dists = [(w.x - extractor.x)**2 + (w.y - extractor.y)**2 for w in workers]
        if sum([w.is_selected for w in workers]) == 1 \
                and all([u[2] > 0 for u in self.obs.observation.single_select]) \
                and len(self.obs.observation.single_select) == 1:
            selected = [w for w in workers if w.is_selected == 1][0]
            dists.sort()
            if (selected.x - extractor.x)**2 + (selected.y - extractor.y)**2 \
                                                            <= dists[:3][-1]:
                return self.do(RAW_FUNCTIONS.Harvest_Gather_screen,
                                            'now', (extractor.x, extractor.y))

        w = workers[np.argmin(dists)]
        return self.try_select(self.worker_type, how=(w.x, w.y))

    def move_worker_to_minerals(self):
        workers = self.get_units_by_type(self.worker_type)
        workers = [w for w in workers if w.buff_id_0 == 275 and w.order_id_0 == 360]
        if not workers:
            return RAW_FUNCTIONS.no_op()

        if sum([w.is_selected for w in workers]) == 1 \
                and all([u[2] > 0 for u in self.obs.observation.single_select]) \
                and len(self.obs.observation.single_select) == 1:
            selected = [w for w in workers if w.is_selected == 1][0]
            minerals = self.get_units_by_type(units.Neutral.MineralField)
            dists = [(m.x - selected.x)**2 + (m.y - selected.y)**2 for m in minerals]
            m = minerals[np.argmin(dists)]
            return self.do(RAW_FUNCTIONS.Harvest_Gather_screen,
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
                    return self.build_hatchery()
                else:
                    return RAW_FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 14 Extractor
        if not self.unit_count(units.Zerg.Extractor) >= 1:
            if n_workers >= 14:
                if minerals >= 25:
                    return self.build_extractor()
                else:
                    return RAW_FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 14 Spawning pool
        if not self.unit_count(units.Zerg.SpawningPool) >= 1:
            if n_workers >= 14:
                if minerals >= 200:
                    return self.build(RAW_FUNCTIONS.Build_SpawningPool_pt)
                else:
                    return RAW_FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 14 Overlord
        if not self.unit_count_with_training(units.Zerg.Overlord) >= 2:
            if n_workers >= 14:
                if minerals >= 100:
                    return self.train(RAW_FUNCTIONS.Train_Overlord_quick)
                else:
                    return RAW_FUNCTIONS.no_op()
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
                return RAW_FUNCTIONS.no_op()

        # Move drones back to minerals once 100 vespene are harvested
        if not (sum([e.assigned_harvesters for e in extractors]) <= 0 \
                    or (vespene < 100 and not metabolic_boost_researched)):
            return self.try_move_worker_to_minerals()

        # 14 Research metabolic boost
        if not metabolic_boost_researched:
            if n_workers >= 14:
                if minerals >= 100 and vespene >= 100:
                    return self.train(RAW_FUNCTIONS.Research_ZerglingMetabolicBoost_quick,
                                        trainer=units.Zerg.SpawningPool)
                else:
                    return RAW_FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 16 Queen
        if not self.unit_count_with_training(units.Zerg.Queen, trainer=units.Zerg.Hatchery) >= 1:
            if n_workers >= 16:
                if minerals >= 150:
                    return self.train(RAW_FUNCTIONS.Train_Queen_quick,
                                        trainer=units.Zerg.Hatchery)
                else:
                    return RAW_FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 18 Zergling x5 ==> 23
        if not pop >= 23:
            if n_workers >= 16:
                if minerals >= 50:
                    return self.train(RAW_FUNCTIONS.Train_Zergling_quick)
                else:
                    return RAW_FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 23 Overlord
        if not self.unit_count_with_training(units.Zerg.Overlord) >= 3:
            if n_workers >= 16:
                if minerals >= 100:
                    return self.train(RAW_FUNCTIONS.Train_Overlord_quick)
                else:
                    return RAW_FUNCTIONS.no_op()
            else:
                return self.wait_train_drone()

        # 23 Zergling x11 ==> 34
        if not pop >= 34:
            if n_workers >= 16:
                if minerals >= 50:
                    return self.train(RAW_FUNCTIONS.Train_Zergling_quick)
                else:
                    return RAW_FUNCTIONS.no_op()
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

        # action = None ==> plus haut niveau
