import os
import numpy as np
import random as rd
import itertools
import collections
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
        self.enemy_coordinates
        self.dists_first_base = None
        self.memory = Memory()

    def reset(self):
        super().reset()
        if self.log is not None:
            self.log.close()
        self.log = open(f'{os.path.abspath(os.getcwd())}'
                        f'\\data\\logs\\log_{self.__class__.__name__}_game_{self.episodes-1}.txt', 'w')
        self.game_step = 0
        self.dists_first_base = None
        self.memory = Memory()

    def step(self, obs):
        super().step(obs)
        self.obs = obs
        self.action = None
        self.memory.update(self.obs)

        if obs.first():
            self.log.write('### Game log ###')
            # Get enemy spawn coordinates
            hatchery = self.get_bases()[0]
            self.enemy_coordinates = (RESOLUTION - 1 - hatchery.x, RESOLUTION - 1 - hatchery.y)

        # Log data
        self.log.write(f'\n\nTime step {self.game_step}')
        self.log.write(f'\nResources : {self.obs.observation.player.minerals}, '
                       f'{self.obs.observation.player.vespene}')
        self.log.write(f'\nSupply : {self.obs.observation.player.food_used}'
                       f'/{self.obs.observation.player.food_cap}')
        self.log.write(f'\nLarvae : {self.unit_count_by_type(units.Zerg.Larva)}')

        self.game_step += 1

    # Core methods
    def unit_count_by_type(self, unit_type):
        return len(self.get_units_by_type(unit_type))

    def get_units_by_type(self, unit_type, from_raw=False):
        if from_raw:
            return [u for u in self.obs.observation.raw_units if u.unit_type == unit_type]
        if unit_type in self.memory.self_units:
            return self.memory.self_units[unit_type]
        if unit_type in self.memory.neutral_units:
            return self.memory.neutral_units[unit_type]
        return []

    def get_bases(self):
        return self.get_units_by_type(units.Zerg.Hatchery) \
             + self.get_units_by_type(units.Zerg.Lair) \
             + self.get_units_by_type(units.Zerg.Hive) \
             + self.get_units_by_type(units.Protoss.Nexus) \
             + self.get_units_by_type(units.Terran.CommandCenter) \
             + self.get_units_by_type(units.Terran.CommandCenterFlying) \
             + self.get_units_by_type(units.Terran.OrbitalCommand) \
             + self.get_units_by_type(units.Terran.OrbitalCommandFlying) \
             + self.get_units_by_type(units.Terran.PlanetaryFortress)

    def get_workers(self):
        return self.get_units_by_type(units.Zerg.Drone) \
             + self.get_units_by_type(units.Protoss.Probe) \
             + self.get_units_by_type(units.Terran.SCV)

    def get_minerals(self, from_raw=False):
        return list(itertools.chain.from_iterable(
            [self.get_units_by_type(getattr(units.Neutral, attr), from_raw=from_raw)
                for attr in dir(units.Neutral) if 'MineralField' in attr]))
                
    def get_vespenes(self, from_raw=False):
        vespenes = list(itertools.chain.from_iterable(
            [self.get_units_by_type(getattr(units.Neutral, attr), from_raw=from_raw)
                for attr in dir(units.Neutral) if 'Geyser' in attr]))
        extractors = self.get_units_by_type(units.Zerg.Extractor) \
                   + self.get_units_by_type(units.Zerg.ExtractorRich) \
                   + self.get_units_by_type(units.Protoss.Assimilator) \
                   + self.get_units_by_type(units.Protoss.AssimilatorRich) \
                   + self.get_units_by_type(units.Terran.Refinery) \
                   + self.get_units_by_type(units.Terran.RefineryRich)
        neutral_vespenes = [v for v in vespenes if all([(v.x, v.y) != (e.x, e.y) for e in extractors])]
        return neutral_vespenes

    def do(self, action, *args, raise_error=True):
        self.log.write(f'\nChose action {action}, args {list(args)}')
        return action(*args)

    # Base actions
    def build(self, action, builder=None, how=1, where='random'):
        ################################################### energy/cooldown (tumor/queen) + how='creep'
        ################################################### nydus worm
        if isinstance(builder, list):
            builders = list(itertools.chain.from_iterable(
                [self.get_units_by_type(unit_type) for unit_type in builder]))
        elif builder is None:
            builders = self.get_workers()
        else:
            builders = self.get_units_by_type(builder)
        builders_available = [b for b in builders if b.order_id_0 == RAW_FUNCTIONS.no_op.id]
        if not builders_available:
            builders_available = [b for b in builders if b.order_id_0 in
                                   [RAW_FUNCTIONS.Harvest_Gather_unit.id,
                                    RAW_FUNCTIONS.Harvest_Gather_Drone_unit.id,
                                    RAW_FUNCTIONS.Harvest_Gather_Mule_unit.id,
                                    RAW_FUNCTIONS.Harvest_Gather_Probe_unit.id,
                                    RAW_FUNCTIONS.Harvest_Gather_SCV_unit.id,
                                    RAW_FUNCTIONS.Harvest_Return_quick.id,
                                    RAW_FUNCTIONS.Harvest_Return_Drone_quick.id,
                                    RAW_FUNCTIONS.Harvest_Return_Mule_quick.id,
                                    RAW_FUNCTIONS.Harvest_Return_Probe_quick.id,
                                    RAW_FUNCTIONS.Harvest_Return_SCV_quick.id,]]
        builders = builders_available
        if not builders and (how == 'base' or how < 50):
            return None # keep looking for an action to take

        # Choose target location / tag
        if where == 'random': # tag target cannot be random, must be deterministic
            # Create the matrix of possible locations
            creep_constraint = np.ones((RESOLUTION, RESOLUTION))
            if self.race_name == 'zerg' and action != RAW_FUNCTIONS.Build_Hatchery_pt:
                creep_constraint = self.obs.observation.feature_minimap.creep
            player_relative = self.obs.observation.feature_minimap.player_relative
            diam_neutral = int(THRESHOLD * 0.48) # unbuildable border around minerals / vespene
            can_build = np.stack([self.obs.observation.feature_minimap.pathable,
                                  self.obs.observation.feature_minimap.buildable,
                                  # should be improved to not let flying units block :
                                  (player_relative == 0).astype('uint8'),
                                  cv2.erode(np.array(player_relative != 3).astype('uint8'),
                                            np.ones((diam_neutral, diam_neutral))),
                                  creep_constraint]).all(axis=0).T.astype('uint8')
            diam = int(THRESHOLD * 0.23) # approximate standard building size
            if action in [RAW_FUNCTIONS.Build_CommandCenter_pt,
                          RAW_FUNCTIONS.Build_Nexus_pt,
                          RAW_FUNCTIONS.Build_Hatchery_pt]:
                diam = int(THRESHOLD * 0.38) # approximate base building size
            can_build = cv2.erode(can_build, np.ones((diam, diam)))

            # Choose the marker point
            if how == 'base':
                minerals = self.get_minerals(from_raw=True)
                bases = self.get_bases()

                def shortest_path(grid, targets, max_lookup):
                    wall = 0 # wall value in the pathable matrix
                    dist_matrix = np.zeros_like(grid).astype(float) - 1
                    dist = 0
                    for t in targets:
                        dist_matrix[t] = dist
                    edge = targets
                    while edge and dist < max_lookup:
                        new_edge = []
                        for x, y in edge:
                            for x2, y2 in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                                # Not a wall and not seen :
                                if 0 <= x2 < RESOLUTION and 0 <= y2 < RESOLUTION \
                                        and grid[x2, y2] != wall and dist_matrix[x2, y2] < 0:
                                    dist_matrix[x2, y2] = dist + 1
                                    new_edge.append((x2, y2))
                        dist += 1
                        edge = new_edge
                    dist_matrix[dist_matrix == -1] = np.inf
                    dist_matrix = np.maximum(dist_matrix, int(diam_neutral / 2))
                    return dist_matrix

                # Select a marker point x, y :
                pathable = np.stack([self.obs.observation.feature_minimap.pathable.T,
                                     (self.obs.observation.feature_minimap.player_id > 0).T]).any(axis=0)
                if self.dists_first_base is None: # only computed once
                    self.dists_first_base = shortest_path(pathable, [self.memory.base_locations[0]],
                                                                            max_lookup=RESOLUTION**2)
                dists_all_bases = [[(m.x - b.x)**2 + (m.y - b.y)**2 for b in bases] for m in minerals]
                if bases:
                    dists_all_bases = np.min(np.sqrt(dists_all_bases), axis=1)
                else:
                    dists_all_bases = np.zeros(len(minerals)) + np.inf
                minerals = [m for i, m in enumerate(minerals) if dists_all_bases[i] > THRESHOLD]
                dists_first_base = [self.dists_first_base[m.x, m.y] for m in minerals]
                chosen_mineral = minerals[np.argmin(dists_first_base)]
                x, y = chosen_mineral.x, chosen_mineral.y

                # Build a score for proximity to minerals :
                minerals_vespenes = self.get_minerals(from_raw=True) + self.get_vespenes(from_raw=True)
                dists = [(m.x - x)**2 + (m.y - y)**2 for m in minerals_vespenes]
                # Select minerals and vespenes in the line :
                minerals_vespenes = [minerals_vespenes[i] for i in np.argsort(dists)[:10]]
                x, y = np.median([[m.x, m.y] for m in minerals_vespenes], axis=0).astype(int)
                path_dists = shortest_path(pathable, [(x, y)], max_lookup=int(THRESHOLD * 1.5))
                best_spots = np.exp(-THRESHOLD * 0.0025 * path_dists)
            else:
                base = how if how < 50 else 1
                x, y = self.memory.base_locations[base - 1]
                best_spots = None # use closest

            # Find the best location near the marker point
            n = int(THRESHOLD * 2)
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

        # Choose builder
        if how == 'base':
            x, y = target if where == 'random' else self.memory.base_locations[max(len(self.get_bases()) - 1, 0)]
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

        return self.do(action, 'now', tag, target)

    def train(self, action, trainer, how='random'):
        if isinstance(trainer, list):
            trainers = list(itertools.chain.from_iterable(
                [self.get_units_by_type(unit_type) for unit_type in trainer]))
        else:
            trainers = self.get_units_by_type(trainer)
        trainers_available = [t for t in trainers if t.order_id_0 == RAW_FUNCTIONS.no_op.id]
        if not trainers_available:
            trainers_available = [t for t in trainers if t.order_id_0 in
                                                    [RAW_FUNCTIONS.Move_pt.id,
                                                     RAW_FUNCTIONS.Move_unit.id,
                                                     RAW_FUNCTIONS.Move_Move_pt.id,
                                                     RAW_FUNCTIONS.Move_Move_unit.id,
                                                     RAW_FUNCTIONS.Attack_pt.id,
                                                     RAW_FUNCTIONS.Attack_unit.id,
                                                     RAW_FUNCTIONS.Attack_Attack_pt.id,
                                                     RAW_FUNCTIONS.Attack_Attack_unit.id,
                                                     RAW_FUNCTIONS.Attack_AttackBuilding_pt.id,
                                                     RAW_FUNCTIONS.Attack_AttackBuilding_unit.id,
                                                     RAW_FUNCTIONS.Attack_Redirect_pt.id,
                                                     RAW_FUNCTIONS.Attack_Redirect_unit.id]]
        trainers = trainers_available

        if how == 'random':
            if not trainers:
                return None # keep looking for an action to take
            picked = rd.choice(trainers)
            tag = picked.tag
        elif how < 50: # how is the number of the base to train from
            if not trainers:
                return None # keep looking for an action to take
            x, y = self.memory.base_locations[how - 1]
            dists = [(t.x - x)**2 + (t.y - y)**2 for t in trainers]
            picked = trainers[np.argmin(dists)]
            tag = picked.tag
        else: # how is the tag of the trainer
            tag = how

        return self.do(action, 'now', tag)

    def attack(self, attackers, coordinates):
        attackers = [u.tag for u in attackers]
        return self.do(RAW_FUNCTIONS.Attack_minimap, 'now', attackers, coordinates)

    def wait(self):
        return self.do(RAW_FUNCTIONS.no_op)


class ZergAgent(RawAgent):
    def __init__(self):
        super().__init__()
        self.race_name = 'zerg'
        self.worker_type = units.Zerg.Drone

    # Core methods
    def unit_count_agg(self, name, with_burrowed=True, with_training=True):
        units_agg = self.get_units_agg(name, with_burrowed=with_burrowed, with_training=with_training)
        if name == 'Zergling':
            # 2 zerglings per cocoon
            return len(units_agg) + sum([u.unit_type == units.Zerg.Cocoon for u in units_agg])
        else:
            return len(units_agg)

    def get_units_agg(self, name, with_burrowed=True, with_training=False):
        if name == 'Baneling':
            res = self.get_units_by_type(units.Zerg.Baneling)
            if with_burrowed:
                res += self.get_units_by_type(units.Zerg.BanelingBurrowed)
            if with_training:
                res += self.get_units_by_type(units.Zerg.BanelingCocoon)
        elif name == 'BroodLord':
            res = self.get_units_by_type(units.Zerg.BroodLord)
            if with_training:
                res += self.get_units_by_type(units.Zerg.BroodLordCocoon)
        elif name == 'Cocoon':
            res = self.get_units_by_type(units.Zerg.Cocoon)
        elif name == 'Corruptor':
            res = self.get_units_by_type(units.Zerg.Corruptor)
            if with_training:
                res += [u for u in self.get_units_by_type(units.Zerg.Cocoon)
                        if u.order_id_0 == RAW_FUNCTIONS.Train_Corruptor_quick.id]
        elif name == 'Drone':
            res = self.get_units_by_type(units.Zerg.Drone)
            if with_burrowed:
                res += self.get_units_by_type(units.Zerg.DroneBurrowed)
            if with_training:
                res += [u for u in self.get_units_by_type(units.Zerg.Cocoon)
                        if u.order_id_0 == RAW_FUNCTIONS.Train_Drone_quick.id]
        elif name == 'Hydralisk':
            res = self.get_units_by_type(units.Zerg.Hydralisk)
            if with_burrowed:
                res += self.get_units_by_type(units.Zerg.HydraliskBurrowed)
            if with_training:
                res += [u for u in self.get_units_by_type(units.Zerg.Cocoon)
                        if u.order_id_0 == RAW_FUNCTIONS.Train_Hydralisk_quick.id]
        elif name == 'Infestor':
            res = self.get_units_by_type(units.Zerg.Infestor)
            if with_burrowed:
                res += self.get_units_by_type(units.Zerg.InfestorBurrowed)
            if with_training:
                res += [u for u in self.get_units_by_type(units.Zerg.Cocoon)
                        if u.order_id_0 == RAW_FUNCTIONS.Train_Infestor_quick.id]
        elif name == 'Larva':
            res = self.get_units_by_type(units.Zerg.Larva)
        elif name == 'Lurker':
            res = self.get_units_by_type(units.Zerg.Lurker)
            if with_burrowed:
                res += self.get_units_by_type(units.Zerg.LurkerBurrowed)
            if with_training:
                res += self.get_units_by_type(units.Zerg.LurkerCocoon)
        elif name == 'Mutalisk':
            res = self.get_units_by_type(units.Zerg.Mutalisk)
            if with_training:
                res += [u for u in self.get_units_by_type(units.Zerg.Cocoon)
                        if u.order_id_0 == RAW_FUNCTIONS.Train_Mutalisk_quick.id]
        elif name == 'Overlord':
            res = self.get_units_by_type(units.Zerg.Overlord) \
                + self.get_units_by_type(units.Zerg.OverlordTransport) \
                + self.get_units_by_type(units.Zerg.Overseer) \
                + self.get_units_by_type(units.Zerg.OverseerOversightMode)
            if with_training:
                res += [u for u in self.get_units_by_type(units.Zerg.Cocoon)
                        if u.order_id_0 == RAW_FUNCTIONS.Train_Overlord_quick.id] \
                     + self.get_units_by_type(units.Zerg.OverlordTransportCocoon) \
                     + self.get_units_by_type(units.Zerg.OverseerCocoon)
        elif name == 'OverlordTransport':
            res = self.get_units_by_type(units.Zerg.OverlordTransport)
            if with_training:
                res += self.get_units_by_type(units.Zerg.OverlordTransportCocoon)
        elif name == 'Overseer':
            res = self.get_units_by_type(units.Zerg.Overseer) \
                + self.get_units_by_type(units.Zerg.OverseerOversightMode)
            if with_training:
                res += self.get_units_by_type(units.Zerg.OverseerCocoon)
        elif name == 'Queen':
            res = self.get_units_by_type(units.Zerg.Queen)
            if with_burrowed:
                res += self.get_units_by_type(units.Zerg.QueenBurrowed)
            if with_training:
                res += [u for u in self.get_bases()
                        if u.order_id_0 == RAW_FUNCTIONS.Train_Queen_quick.id]
        elif name == 'Ravager':
            res = self.get_units_by_type(units.Zerg.Ravager)
            if with_burrowed:
                res += self.get_units_by_type(units.Zerg.RavagerBurrowed)
            if with_training:
                res += self.get_units_by_type(units.Zerg.RavagerCocoon)
        elif name == 'Roach':
            res = self.get_units_by_type(units.Zerg.Roach)
            if with_burrowed:
                res += self.get_units_by_type(units.Zerg.RoachBurrowed)
            if with_training:
                res += [u for u in self.get_units_by_type(units.Zerg.Cocoon)
                        if u.order_id_0 == RAW_FUNCTIONS.Train_Roach_quick.id]
        elif name == 'SwarmHost':
            res = self.get_units_by_type(units.Zerg.SwarmHost)
            if with_burrowed:
                res += self.get_units_by_type(units.Zerg.SwarmHostBurrowed)
            if with_training:
                res += [u for u in self.get_units_by_type(units.Zerg.Cocoon)
                        if u.order_id_0 == RAW_FUNCTIONS.Train_SwarmHost_quick.id]
        elif name == 'Ultralisk':
            res = self.get_units_by_type(units.Zerg.Ultralisk)
            if with_burrowed:
                res += self.get_units_by_type(units.Zerg.UltraliskBurrowed)
            if with_training:
                res += [u for u in self.get_units_by_type(units.Zerg.Cocoon)
                        if u.order_id_0 == RAW_FUNCTIONS.Train_Ultralisk_quick.id]
        elif name == 'Viper':
            res = self.get_units_by_type(units.Zerg.Viper)
            if with_training:
                res += [u for u in self.get_units_by_type(units.Zerg.Cocoon)
                        if u.order_id_0 == RAW_FUNCTIONS.Train_Viper_quick.id]
        elif name == 'Zergling':
            res = self.get_units_by_type(units.Zerg.Zergling)
            if with_burrowed:
                res += self.get_units_by_type(units.Zerg.ZerglingBurrowed)
            if with_training:
                res += [u for u in self.get_units_by_type(units.Zerg.Cocoon)
                        if u.order_id_0 == RAW_FUNCTIONS.Train_Zergling_quick.id]
        elif name == 'BanelingNest':
            res = self.get_units_by_type(units.Zerg.BanelingNest)
        elif name == 'CreepTumor':
            res = self.get_units_by_type(units.Zerg.CreepTumor) \
                + self.get_units_by_type(units.Zerg.CreepTumorQueen)
            if with_burrowed:
                res += self.get_units_by_type(units.Zerg.CreepTumorBurrowed)
        elif name == 'EvolutionChamber':
            res = self.get_units_by_type(units.Zerg.EvolutionChamber)
        elif name == 'Extractor':
            res = self.get_units_by_type(units.Zerg.Extractor) \
                + self.get_units_by_type(units.Zerg.ExtractorRich)
        elif name == 'Hatchery':
            res = self.get_units_by_type(units.Zerg.Hatchery) \
                + self.get_units_by_type(units.Zerg.Lair) \
                + self.get_units_by_type(units.Zerg.Hive)
        elif name == 'Lair':
            res = self.get_units_by_type(units.Zerg.Lair) \
                + self.get_units_by_type(units.Zerg.Hive)
        elif name == 'Hive':
            res = self.get_units_by_type(units.Zerg.Hive)
        elif name == 'HydraliskDen':
            res = self.get_units_by_type(units.Zerg.HydraliskDen)
        elif name == 'InfestationPit':
            res = self.get_units_by_type(units.Zerg.InfestationPit)
        elif name == 'LurkerDen':
            res = self.get_units_by_type(units.Zerg.LurkerDen)
        elif name == 'NydusCanal':
            res = self.get_units_by_type(units.Zerg.NydusCanal)
        elif name == 'NydusNetwork':
            res = self.get_units_by_type(units.Zerg.NydusNetwork)
        elif name == 'RoachWarren':
            res = self.get_units_by_type(units.Zerg.RoachWarren)
        elif name == 'SpawningPool':
            res = self.get_units_by_type(units.Zerg.SpawningPool)
        elif name == 'SpineCrawler':
            res = self.get_units_by_type(units.Zerg.SpineCrawler) \
                + self.get_units_by_type(units.Zerg.SpineCrawlerUprooted)
        elif name == 'Spire':
            res = self.get_units_by_type(units.Zerg.Spire) \
                + self.get_units_by_type(units.Zerg.GreaterSpire)
        elif name == 'GreaterSpire':
            res = self.get_units_by_type(units.Zerg.GreaterSpire)
        elif name == 'SporeCrawler':
            res = self.get_units_by_type(units.Zerg.SporeCrawler) \
                + self.get_units_by_type(units.Zerg.SporeCrawlerUprooted)
        elif name == 'UltraliskCavern':
            res = self.get_units_by_type(units.Zerg.UltraliskCavern)
        elif name == 'Burrow':
            res = [True] if upgrades.Upgrades.Burrow in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('Hatchery')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_Burrow_quick.id]
        elif name == 'PneumatizedCarapace':
            res = [True] if upgrades.Upgrades.PneumatizedCarapace in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('Hatchery')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_PneumatizedCarapace_quick.id]
        elif name == 'MetabolicBoost':
            res = [True] if upgrades.Upgrades.MetabolicBoost in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('SpawningPool')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZerglingMetabolicBoost_quick.id]
        elif name == 'MeleeAttack1':
            res = [True] if upgrades.Upgrades.ZergMeleeWeaponsLevel1 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergMeleeWeaponsLevel2 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergMeleeWeaponsLevel3 in self.obs.observation.upgrades \
                            else []
            if with_training:
                res += [u for u in self.get_units_agg('EvolutionChamber')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergMeleeWeaponsLevel1_quick.id]
        elif name == 'MeleeAttack2':
            res = [True] if upgrades.Upgrades.ZergMeleeWeaponsLevel2 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergMeleeWeaponsLevel3 in self.obs.observation.upgrades \
                            else []
            if with_training:
                res += [u for u in self.get_units_agg('EvolutionChamber')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergMeleeWeaponsLevel2_quick.id]
        elif name == 'MeleeAttack3':
            res = [True] if upgrades.Upgrades.ZergMeleeWeaponsLevel3 in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('EvolutionChamber')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergMeleeWeaponsLevel3_quick.id]
        elif name == 'MissileAttack1':
            res = [True] if upgrades.Upgrades.ZergMissileWeaponsLevel1 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergMissileWeaponsLevel2 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergMissileWeaponsLevel3 in self.obs.observation.upgrades \
                            else []
            if with_training:
                res += [u for u in self.get_units_agg('EvolutionChamber')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergMissileWeaponsLevel1_quick.id]
        elif name == 'MissileAttack2':
            res = [True] if upgrades.Upgrades.ZergMissileWeaponsLevel2 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergMissileWeaponsLevel3 in self.obs.observation.upgrades \
                            else []
            if with_training:
                res += [u for u in self.get_units_agg('EvolutionChamber')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergMissileWeaponsLevel2_quick.id]
        elif name == 'MissileAttack3':
            res = [True] if upgrades.Upgrades.ZergMissileWeaponsLevel3 in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('EvolutionChamber')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergMissileWeaponsLevel3_quick.id]
        elif name == 'GroundArmor1':
            res = [True] if upgrades.Upgrades.ZergGroundArmorsLevel1 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergGroundArmorsLevel2 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergGroundArmorsLevel3 in self.obs.observation.upgrades \
                            else []
            if with_training:
                res += [u for u in self.get_units_agg('EvolutionChamber')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergGroundArmorLevel1_quick.id]
        elif name == 'GroundArmor2':
            res = [True] if upgrades.Upgrades.ZergGroundArmorsLevel2 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergGroundArmorsLevel3 in self.obs.observation.upgrades \
                            else []
            if with_training:
                res += [u for u in self.get_units_agg('EvolutionChamber')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergGroundArmorLevel2_quick.id]
        elif name == 'GroundArmor3':
            res = [True] if upgrades.Upgrades.ZergGroundArmorsLevel3 in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('EvolutionChamber')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergGroundArmorLevel3_quick.id]
        elif name == 'FlyerAttack1':
            res = [True] if upgrades.Upgrades.ZergFlyerWeaponsLevel1 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergFlyerWeaponsLevel2 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergFlyerWeaponsLevel3 in self.obs.observation.upgrades \
                            else []
            if with_training:
                res += [u for u in self.get_units_agg('Spire')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergFlyerAttackLevel1_quick.id]
        elif name == 'FlyerAttack2':
            res = [True] if upgrades.Upgrades.ZergFlyerWeaponsLevel2 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergFlyerWeaponsLevel3 in self.obs.observation.upgrades \
                            else []
            if with_training:
                res += [u for u in self.get_units_agg('Spire')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergFlyerAttackLevel2_quick.id]
        elif name == 'FlyerAttack3':
            res = [True] if upgrades.Upgrades.ZergFlyerWeaponsLevel3 in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('Spire')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergFlyerAttackLevel3_quick.id]
        elif name == 'FlyerArmor1':
            res = [True] if upgrades.Upgrades.ZergFlyerArmorsLevel1 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergFlyerArmorsLevel2 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergFlyerArmorsLevel3 in self.obs.observation.upgrades \
                            else []
            if with_training:
                res += [u for u in self.get_units_agg('Spire')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergFlyerArmorLevel1_quick.id]
        elif name == 'FlyerArmor2':
            res = [True] if upgrades.Upgrades.ZergFlyerArmorsLevel2 in self.obs.observation.upgrades \
                            or upgrades.Upgrades.ZergFlyerArmorsLevel3 in self.obs.observation.upgrades \
                            else []
            if with_training:
                res += [u for u in self.get_units_agg('Spire')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergFlyerArmorLevel2_quick.id]
        elif name == 'FlyerArmor3':
            res = [True] if upgrades.Upgrades.ZergFlyerArmorsLevel3 in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('Spire')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZergFlyerArmorLevel3_quick.id]
        elif name == 'MuscularAugments':
            res = [True] if upgrades.Upgrades.MuscularAugments in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('HydraliskDen')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_MuscularAugments_quick.id]
        elif name == 'GlialReconstitution':
            res = [True] if upgrades.Upgrades.GlialReconstitution in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('RoachWarren')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_GlialRegeneration_quick.id]
        elif name == 'TunnelingClaws':
            res = [True] if upgrades.Upgrades.TunnelingClaws in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('RoachWarren')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_TunnelingClaws_quick.id]
        elif name == 'CentrifugalHooks':
            res = [True] if upgrades.Upgrades.CentrificalHooks in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('BanelingNest')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_CentrifugalHooks_quick.id]
        elif name == 'NeuralParasite':
            res = [True] if upgrades.Upgrades.NeuralParasite in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('InfestationPit')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_NeuralParasite_quick.id]
        elif name == 'PathogenGlands':
            res = [True] if upgrades.Upgrades.PathogenGlands in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('InfestationPit')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_PathogenGlands_quick.id]
        elif name == 'GroovedSpines':
            res = [True] if upgrades.Upgrades.GroovedSpines in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('HydraliskDen')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_GroovedSpines_quick.id]
        elif name == 'AdrenalGlands':
            res = [True] if upgrades.Upgrades.AdrenalGlands in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('SpawningPool')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ZerglingAdrenalGlands_quick.id]
        elif name == 'ChitinousPlating':
            res = [True] if upgrades.Upgrades.ChitinousPlating in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('UltraliskCavern')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_ChitinousPlating_quick.id]
        elif name == 'AnabolicSynthesis':
            res = [True] if upgrades.Upgrades.AnabolicSynthesis in self.obs.observation.upgrades else []
            if with_training:
                res += [u for u in self.get_units_agg('UltraliskCavern')
                        if u.order_id_0 == RAW_FUNCTIONS.Research_AnabolicSynthesis_quick.id]
        else:
            raise Warning(f'Unknown unit name {name}')

        if not with_training:
            res = [u for u in res if u.build_progress == 100]

        return res

    def check_requirements(self, req, n=1, how=1):
        if req == 'BanelingNest':
            return self.check_requirements_building('BanelingNest', self.build_baneling_nest)
        elif req == 'CreepTumor':
            return self.check_requirements_building('CreepTumor', self.build_creep_tumor_queen)
        elif req == 'GreaterSpire':
            return self.check_requirements_building('GreaterSpire', self.build_greater_spire)
        elif req == 'Hatchery':
            return self.check_requirements_building('Hatchery', self.build_hatchery)
        elif req == 'HatcheryExact':
            hatcheries = self.get_units_by_type(units.Zerg.Hatchery)
            if len(hatcheries) == 0:
                return self.build_hatchery()
            elif len([h for h in hatcheries if h.build_progress == 100]) == 0:
                return False
            return True
        elif req == 'Lair':
            return self.check_requirements_building('Lair', self.build_lair)
        elif req == 'LairExact':
            lairs = self.get_units_by_type(units.Zerg.Lair)
            if len(lairs) == 0:
                return self.build_lair()
            elif len([l for l in lairs if l.build_progress == 100]) == 0:
                return False
            return True
        elif req == 'Hive':
            return self.check_requirements_building('Hive', self.build_hive)
        elif req == 'HydraliskDen':
            return self.check_requirements_building('HydraliskDen', self.build_hydralisk_den)
        elif req == 'InfestationPit':
            return self.check_requirements_building('InfestationPit', self.build_infestation_pit)
        elif req == 'LurkerDen':
            return self.check_requirements_building('LurkerDen', self.build_lurker_den)
        elif req == 'NydusNetwork':
            return self.check_requirements_building('NydusNetwork', self.build_nydus_network)
        elif req == 'RoachWarren':
            return self.check_requirements_building('RoachWarren', self.build_roach_warren)
        elif req == 'SpawningPool':
            return self.check_requirements_building('SpawningPool', self.build_spawning_pool)
        elif req == 'Spire':
            return self.check_requirements_building('Spire', self.build_spire)
        elif req == 'UltraliskCavern':
            return self.check_requirements_building('UltraliskCavern', self.build_ultralisk_cavern)
        elif req == 'MeleeAttack1':
            return self.check_requirements_building('MeleeAttack1', self.research_melee_attack_1)
        elif req == 'MeleeAttack2':
            return self.check_requirements_building('MeleeAttack2', self.research_melee_attack_2)
        elif req == 'MissileAttack1':
            return self.check_requirements_building('MissileAttack1', self.research_missile_attack_1)
        elif req == 'MissileAttack2':
            return self.check_requirements_building('MissileAttack2', self.research_missile_attack_2)
        elif req == 'GroundArmor1':
            return self.check_requirements_building('GroundArmor1', self.research_ground_armor_1)
        elif req == 'GroundArmor2':
            return self.check_requirements_building('GroundArmor2', self.research_ground_armor_2)
        elif req == 'FlyerAttack1':
            return self.check_requirements_building('FlyerAttack1', self.research_flyer_attack_1)
        elif req == 'FlyerAttack2':
            return self.check_requirements_building('FlyerAttack2', self.research_flyer_attack_2)
        elif req == 'FlyerArmor1':
            return self.check_requirements_building('FlyerArmor1', self.research_flyer_armor_1)
        elif req == 'FlyerArmor2':
            return self.check_requirements_building('FlyerArmor2', self.research_flyer_armor_2)
        elif req == 'Corruptor':
            return self.check_requirements_unit('Corruptor', self.train_corruptor, n=n, how=how)
        elif req == 'Hydralisk':
            return self.check_requirements_unit('Hydralisk', self.train_hydralisk, n=n, how=how)
        elif req == 'Queen':
            return self.check_requirements_unit('Queen', self.train_queen, n=n, how=how)
        elif req == 'Roach':
            return self.check_requirements_unit('Roach', self.train_roach, n=n, how=how)
        elif req == 'Zergling':
            return self.check_requirements_unit('Zergling', self.train_zergling, n=n, how=how)
        else:
            raise Warning(f'Unknown unit name {req}')

    def check_requirements_building(self, req, action):
        if self.unit_count_agg(req, with_burrowed=False) == 0:
            return action()
        elif self.unit_count_agg(req, with_burrowed=False, with_training=False) == 0:
            return False
        return True

    def check_requirements_unit(self, req, action, n=1, how='random'):
        if self.unit_count_agg(req, with_training=False, with_burrowed=False) == 0:
            if self.unit_count_agg(req, with_burrowed=False) < n:
                return action(how=how)
            return False
        return True

    # Specific actions
    def train_check_conditions(self, action, trainer=units.Zerg.Larva,
                               requirements=None, n=1,
                               minerals=0, vespene=0, supply=0, how='random'):
        if requirements is not None:
            build_states = []
            for req in requirements:
                b_state = self.check_requirements(req, n=n, how=how)
                if not isinstance(b_state, bool):
                    return b_state
                build_states.append(b_state)
            if not all(build_states):
                return None

        enough_minerals = self.obs.observation.player.minerals >= minerals
        enough_vespene = self.obs.observation.player.vespene >= vespene
        enough_supply = self.obs.observation.player.food_cap \
                        >= self.obs.observation.player.food_used + supply
        if not isinstance(trainer, list):
            trainer = [trainer]
        available_trainers = sum([self.unit_count_by_type(t) for t in trainer]) > 0
        if enough_minerals and enough_vespene and enough_supply and available_trainers:
            return self.train(action, trainer, how=how)
        elif not enough_supply:
            if self.unit_count_agg('Overlord') == self.unit_count_agg('Overlord', with_training=False):
                return self.train_overlord()
            else:
                return None
        elif not enough_minerals or not enough_vespene:
            return self.wait()
        else: # no available trainers
            return None # keep looking for an action to take

    def build_check_conditions(self, action, builder=units.Zerg.Drone,
                               requirements=None,
                               minerals=0, vespene=0, how=1, where='random'):
        if requirements is not None:
            build_states = []
            for req in requirements:
                b_state = self.check_requirements(req, how=how)
                if not isinstance(b_state, bool):
                    return b_state
                build_states.append(b_state)
            if not all(build_states):
                return None

        enough_minerals = self.obs.observation.player.minerals >= minerals
        enough_vespene = self.obs.observation.player.vespene >= vespene
        if not isinstance(builder, list):
            builder = [builder]
        available_builders = sum([self.unit_count_by_type(b) for b in builder]) > 0
        if enough_minerals and enough_vespene and available_builders:
            return self.build(action, builder, how=how, where=where)
        elif not enough_minerals or not enough_vespene:
            return self.wait()
        else: # no available builders
            return None # keep looking for an action to take

    def train_baneling(self, n=1, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_Baneling_quick,
                trainer=units.Zerg.Zergling,
                requirements=['BanelingNest', 'Zergling'], n=n,
                minerals=25, vespene=25, supply=1, how=how)

    def train_brood_lord(self, n=1, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Morph_BroodLord_quick,
                trainer=units.Zerg.Corruptor,
                requirements=['GreaterSpire', 'Corruptor'], n=n,
                minerals=150, vespene=150, supply=2, how=how)

    def train_corruptor(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_Corruptor_quick,
                requirements=['Spire'],
                minerals=150, vespene=100, supply=2, how=how)

    def train_drone(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_Drone_quick,
                requirements=['Hatchery'],
                minerals=50, supply=1, how=how)

    def train_hydralisk(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_Hydralisk_quick,
                requirements=['HydraliskDen'],
                minerals=100, vespene=50, supply=2, how=how)

    def train_infestor(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_Infestor_quick,
                requirements=['InfestationPit'],
                minerals=100, vespene=150, supply=2, how=how)

    def train_lurker(self, n=1, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Morph_Lurker_quick,
                requirements=['LurkerDen', 'Hydralisk'], n=n,
                trainer=units.Zerg.Hydralisk,
                minerals=50, vespene=100, supply=3, how=how)

    def train_mutalisk(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_Mutalisk_quick,
                requirements=['Spire'],
                minerals=100, vespene=100, supply=2, how=how)

    def train_overlord(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_Overlord_quick,
                requirements=['Hatchery'],
                minerals=100, how=how)

    def train_overlord_transport(self, n=1, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Morph_OverlordTransport_quick,
                trainer=units.Zerg.Overlord,
                requirements=['Lair', 'Overlord'], n=n,
                minerals=25, vespene=25, how=how)

    def train_overseer(self, n=1, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Morph_Overseer_quick,
                trainer=units.Zerg.Overlord,
                requirements=['Lair', 'Overlord'], n=n,
                minerals=50, vespene=50, how=how)

    def train_queen(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_Queen_quick,
                trainer=[units.Zerg.Hatchery, units.Zerg.Lair, units.Zerg.Hive],
                requirements=['SpawningPool', 'Hatchery'],
                minerals=150, supply=2, how=how)

    def train_ravager(self, n=1, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Morph_Ravager_quick,
                trainer=units.Zerg.Roach,
                requirements=['RoachWarren', 'Roach'], n=n,
                minerals=25, vespene=75, supply=3, how=how)

    def train_roach(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_Roach_quick,
                requirements=['RoachWarren'],
                minerals=75, vespene=25, supply=2, how=how)

    def train_swarm_host(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_SwarmHost_quick,
                requirements=['InfestationPit'],
                minerals=100, vespene=75, supply=3, how=how)

    def train_ultralisk(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_Ultralisk_quick,
                requirements=['UltraliskCavern'],
                minerals=300, vespene=200, supply=6, how=how)

    def train_viper(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_Viper_quick,
                requirements=['Hive'],
                minerals=100, vespene=200, supply=3, how=how)

    def train_zergling(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Train_Zergling_quick,
                requirements=['SpawningPool'],
                minerals=50, supply=1, how=how)

    def build_greater_spire(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Morph_GreaterSpire_quick,
                trainer=units.Zerg.Spire,
                requirements=['Spire', 'Hive'],
                minerals=100, vespene=150, how=how)

    def build_lair(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Morph_Lair_quick,
                trainer=units.Zerg.Hatchery,
                requirements=['SpawningPool', 'HatcheryExact'],
                minerals=150, vespene=100, how=how)

    def build_hive(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Morph_Hive_quick,
                trainer=units.Zerg.Lair,
                requirements=['InfestationPit', 'LairExact'],
                minerals=200, vespene=150, how=how)

    def research_burrow(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_Burrow_quick,
                trainer=[units.Zerg.Hatchery, units.Zerg.Lair, units.Zerg.Hive],
                requirements=['Hatchery'],
                minerals=100, vespene=100, how=how)

    def research_pneumatized_carapace(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_PneumatizedCarapace_quick,
                trainer=[units.Zerg.Hatchery, units.Zerg.Lair, units.Zerg.Hive],
                requirements=['Hatchery'],
                minerals=100, vespene=100, how=how)

    def research_metabolic_boost(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZerglingMetabolicBoost_quick,
                trainer=units.Zerg.SpawningPool,
                requirements=['SpawningPool'],
                minerals=100, vespene=100, how=how)

    def research_melee_attack_1(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergMeleeWeaponsLevel1_quick,
                trainer=units.Zerg.EvolutionChamber,
                requirements=['EvolutionChamber'],
                minerals=100, vespene=100, how=how)

    def research_melee_attack_2(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergMeleeWeaponsLevel2_quick,
                trainer=units.Zerg.EvolutionChamber,
                requirements=['EvolutionChamber', 'Lair', 'MeleeAttack1'],
                minerals=150, vespene=150, how=how)

    def research_melee_attack_3(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergMeleeWeaponsLevel3_quick,
                trainer=units.Zerg.EvolutionChamber,
                requirements=['EvolutionChamber', 'Hive', 'MeleeAttack2'],
                minerals=200, vespene=200, how=how)

    def research_missile_attack_1(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergMissileWeaponsLevel1_quick,
                trainer=units.Zerg.EvolutionChamber,
                requirements=['EvolutionChamber'],
                minerals=100, vespene=100, how=how)

    def research_missile_attack_2(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergMissileWeaponsLevel2_quick,
                trainer=units.Zerg.EvolutionChamber,
                requirements=['EvolutionChamber', 'Lair', 'MissileAttack1'],
                minerals=150, vespene=150, how=how)

    def research_missile_attack_3(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergMissileWeaponsLevel3_quick,
                trainer=units.Zerg.EvolutionChamber,
                requirements=['EvolutionChamber', 'Hive', 'MissileAttack2'],
                minerals=200, vespene=200, how=how)

    def research_ground_armor_1(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergGroundArmorLevel1_quick,
                trainer=units.Zerg.EvolutionChamber,
                requirements=['EvolutionChamber'],
                minerals=150, vespene=150, how=how)

    def research_ground_armor_2(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergGroundArmorLevel2_quick,
                trainer=units.Zerg.EvolutionChamber,
                requirements=['EvolutionChamber', 'Lair', 'GroundArmor1'],
                minerals=225, vespene=225, how=how)

    def research_ground_armor_3(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergGroundArmorLevel3_quick,
                trainer=units.Zerg.EvolutionChamber,
                requirements=['EvolutionChamber', 'Hive', 'GroundArmor2'],
                minerals=300, vespene=300, how=how)

    def research_flyer_attack_1(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergFlyerAttackLevel1_quick,
                trainer=[units.Zerg.Spire, units.Zerg.GreaterSpire],
                requirements=['Spire'],
                minerals=100, vespene=100, how=how)

    def research_flyer_attack_2(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergFlyerAttackLevel2_quick,
                trainer=[units.Zerg.Spire, units.Zerg.GreaterSpire],
                requirements=['Spire', 'Lair', 'FlyerAttack1'],
                minerals=175, vespene=175, how=how)

    def research_flyer_attack_3(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergFlyerAttackLevel3_quick,
                trainer=[units.Zerg.Spire, units.Zerg.GreaterSpire],
                requirements=['Spire', 'Hive', 'FlyerAttack2'],
                minerals=250, vespene=250, how=how)

    def research_flyer_armor_1(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergFlyerArmorLevel1_quick,
                trainer=[units.Zerg.Spire, units.Zerg.GreaterSpire],
                requirements=['Spire'],
                minerals=150, vespene=150, how=how)

    def research_flyer_armor_2(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergFlyerArmorLevel2_quick,
                trainer=[units.Zerg.Spire, units.Zerg.GreaterSpire],
                requirements=['Spire', 'Lair', 'FlyerArmor1'],
                minerals=225, vespene=225, how=how)

    def research_flyer_armor_3(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZergFlyerArmorLevel3_quick,
                trainer=[units.Zerg.Spire, units.Zerg.GreaterSpire],
                requirements=['Spire', 'Hive', 'FlyerArmor2'],
                minerals=300, vespene=300, how=how)

    def research_muscular_augments(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_MuscularAugments_quick,
                trainer=units.Zerg.HydraliskDen,
                requirements=['HydraliskDen', 'Lair'],
                minerals=100, vespene=100, how=how)

    def research_glial_reconstitution(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_GlialRegeneration_quick,
                trainer=units.Zerg.RoachWarren,
                requirements=['RoachWarren', 'Lair'],
                minerals=100, vespene=100, how=how)

    def research_tunneling_claws(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_TunnelingClaws_quick,
                trainer=units.Zerg.RoachWarren,
                requirements=['RoachWarren', 'Lair'],
                minerals=150, vespene=150, how=how)

    def research_centrifugal_hooks(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_CentrifugalHooks_quick,
                trainer=units.Zerg.BanelingNest,
                requirements=['BanelingNest', 'Lair'],
                minerals=150, vespene=150, how=how)

    def research_neural_parasite(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_NeuralParasite_quick,
                trainer=units.Zerg.InfestationPit,
                requirements=['InfestationPit'],
                minerals=150, vespene=150, how=how)

    def research_pathogen_glands(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_PathogenGlands_quick,
                trainer=units.Zerg.InfestationPit,
                requirements=['InfestationPit'],
                minerals=150, vespene=150, how=how)

    def research_grooved_spines(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_GroovedSpines_quick,
                trainer=units.Zerg.HydraliskDen,
                requirements=['HydraliskDen'],
                minerals=100, vespene=100, how=how)

    def research_adrenal_glands(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ZerglingAdrenalGlands_quick,
                trainer=units.Zerg.SpawningPool,
                requirements=['SpawningPool', 'Hive'],
                minerals=200, vespene=200, how=how)

    def research_chitinous_plating(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_ChitinousPlating_quick,
                trainer=units.Zerg.UltraliskCavern,
                requirements=['UltraliskCavern'],
                minerals=150, vespene=150, how=how)

    def research_anabolic_synthesis(self, how='random'):
        return self.train_check_conditions(RAW_FUNCTIONS.Research_AnabolicSynthesis_quick,
                trainer=units.Zerg.UltraliskCavern,
                requirements=['UltraliskCavern'],
                minerals=150, vespene=150, how=how)

    def build_baneling_nest(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_BanelingNest_pt,
                requirements=['SpawningPool'],
                minerals=100, vespene=50, how=how, where=where)

    def build_creep_tumor_queen(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_CreepTumor_Queen_pt,
                builder=units.Zerg.Queen,
                requirements=['Queen'],
                how=how, where=where)

    def build_creep_tumor_tumor(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_CreepTumor_Tumor_pt,
                builder=[units.Zerg.CreepTumor, units.Zerg.CreepTumorQueen],
                requirements=['CreepTumor'],
                how=how, where=where)

    def build_extractor(self, how=1, where='random'):
        if where == 'random':
            neutral_vespenes = self.get_vespenes()
            x, y = self.memory.base_locations[how - 1]
            dists = [(v.x - x)**2 + (v.y - y)**2 for v in neutral_vespenes]
            vespene = neutral_vespenes[np.argmin(dists)]
            tag = vespene.tag
        else:
            tag = where
        return self.build_check_conditions(RAW_FUNCTIONS.Build_Extractor_unit,
                minerals=25, how=how, where=tag)

    def build_evolution_chamber(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_EvolutionChamber_pt,
                requirements=['Hatchery'],
                minerals=75, how=how, where=where)

    def build_hatchery(self, how='base', where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_Hatchery_pt,
                minerals=300, how=how, where=where)

    def build_hydralisk_den(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_HydraliskDen_pt,
                requirements=['Lair'],
                minerals=100, vespene=100, how=how, where=where)

    def build_infestation_pit(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_InfestationPit_pt,
                requirements=['Lair'],
                minerals=100, vespene=100, how=how, where=where)

    def build_lurker_den(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_LurkerDen_pt,
                requirements=['HydraliskDen'],
                minerals=100, vespene=150, how=how, where=where)

    def build_nydus_canal(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_NydusWorm_pt,
                builder=units.Zerg.NydusNetwork,
                requirements=['NydusNetwork'],
                minerals=75, vespene=75, how=how, where=where)

    def build_nydus_network(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_NydusNetwork_pt,
                requirements=['Lair'],
                minerals=150, vespene=150, how=how, where=where)

    def build_roach_warren(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_RoachWarren_pt,
                requirements=['SpawningPool'],
                minerals=150, how=how, where=where)

    def build_spawning_pool(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_SpawningPool_pt,
                requirements=['Hatchery'],
                minerals=200, how=how, where=where)

    def build_spine_crawler(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_SpineCrawler_pt,
                requirements=['SpawningPool'],
                minerals=100, how=how, where=where)

    def build_spire(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_Spire_pt,
                requirements=['Lair'],
                minerals=200, vespene=200, how=how, where=where)

    def build_spore_crawler(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_SporeCrawler_pt,
                requirements=['SpawningPool'],
                minerals=75, how=how, where=where)

    def build_ultralisk_cavern(self, how=1, where='random'):
        return self.build_check_conditions(RAW_FUNCTIONS.Build_UltraliskCavern_pt,
                requirements=['Hive'],
                minerals=150, vespene=200, how=how, where=where)

    def move_workers_to(self, to, how='all'): # to can be 'gas' or 'minerals'
        checking = 1 if how == 'all' else how
        done = False
        workers = self.get_workers()
        workers_available = [w for w in workers if w.order_id_0 == RAW_FUNCTIONS.no_op.id]
        if not workers_available:
            workers_available = [w for w in workers
                                 if w.order_id_0 == RAW_FUNCTIONS.Harvest_Return_Drone_quick.id]
            if to == 'gas':
                workers_available = [w for w in workers_available if w.buff_id_0 in
                                        [buffs.Buffs.CarryMineralFieldMinerals,
                                         buffs.Buffs.CarryHighYieldMineralFieldMinerals]]
            elif to == 'minerals':
                workers_available = [w for w in workers_available
                                     if w.buff_id_0 == buffs.Buffs.CarryHarvestableVespeneGeyserGasZerg]
            else:
                raise Warning(f'Unknown worker destination {to}')
        if not workers_available:
            return None
        workers = workers_available

        while not done:
            x, y = self.memory.base_locations[checking - 1]
            if to == 'gas':
                spots = self.get_vespenes() + self.get_units_agg('Extractor', with_training=True)
                dists = [(s.x - x)**2 + (s.y - y)**2 for s in spots]
                spots = [spots[i] for i in np.argsort(dists)[:2]]
                spots = [s for s in spots if s.unit_type in
                            [units.Zerg.Extractor, units.Zerg.ExtractorRich]
                            and s.build_progress == 100]
                for s in spots:
                    if s.assigned_harvesters < 3:
                        dists = [(w.x - s.x)**2 + (w.y - s.y)**2 for w in workers]
                        step_workers = [workers[i] for i in np.argsort(dists)[:3 - s.assigned_harvesters]]
                        step_workers = [w.tag for w in step_workers]
                        return self.do(RAW_FUNCTIONS.Harvest_Gather_unit, 'now', step_workers, s.tag)
            elif to == 'minerals':
                spots = self.get_minerals()
                dists = [(s.x - x)**2 + (s.y - y)**2 for s in spots]
                s = spots[np.argmin(dists)]
                step_workers = [w for w in workers if (w.x - x)**2 + (w.y - y)**2 <= THRESHOLD]
                step_workers = [w.tag for w in step_workers]
                if step_workers:
                    return self.do(RAW_FUNCTIONS.Harvest_Gather_unit, 'now', step_workers, s.tag)
            checking += 1
            done = how != 'all' or checking > len(self.get_bases())
        return None

    def move_workers_to_gas(self, how='all'):
        return self.move_workers_to('gas', how=how)

    def move_workers_to_minerals(self, how='all'):
        return self.move_workers_to('minerals', how=how)

    # High-level actions
    def main_hl(self, action, name, n=1, pop=0, increase=None):
        if increase is None:
            increase = self.train_drone
        if self.action is None:
            if not self.unit_count_agg(name) >= n:
                if self.obs.observation.player.food_used >= pop:
                    self.action = action()
                else:
                    self.action = increase()

    ######################################################################################## ADD
    def train_brood_lord_hl(self, n, pop=0, increase=None, how='random'):
        action = lambda : self.train_brood_lord(n=n-self.unit_count_agg('BroodLord'), how=how)
        name = 'BroodLord'
        self.main_hl(action, name, n=n, pop=pop, increase=increase)

    def train_drone_hl(self, n, pop=0, increase=None, how='random'):
        action = lambda : self.train_drone(how=how)
        name = 'Drone'
        self.main_hl(action, name, n=n, pop=pop, increase=increase)

    def train_infestor_hl(self, n, pop=0, increase=None, how='random'):
        action = lambda : self.train_infestor(how=how)
        name = 'Infestor'
        self.main_hl(action, name, n=n, pop=pop, increase=increase)

    def train_overlord_hl(self, n, pop=0, increase=None, how='random'):
        action = lambda : self.train_overlord(how=how)
        name = 'Overlord'
        self.main_hl(action, name, n=n, pop=pop, increase=increase)

    def train_viper_hl(self, n, pop=0, increase=None, how='random'):
        action = lambda : self.train_viper(how=how)
        name = 'Viper'
        self.main_hl(action, name, n=n, pop=pop, increase=increase)

    def train_zergling_hl(self, n, pop=0, increase=None, how='random'):
        action = lambda : self.train_zergling(how=how)
        name = 'Zergling'
        self.main_hl(action, name, n=n, pop=pop, increase=increase)

    def build_extractor_hl(self, n, pop=0, increase=None, how=1):
        action = lambda : self.build_extractor(how=how)
        name = 'Extractor'
        self.main_hl(action, name, n=n, pop=pop, increase=increase)

    def build_hatchery_hl(self, n, pop=0, increase=None, where='random'):
        action = lambda : self.build_hatchery(where=where)
        name = 'Hatchery'
        self.main_hl(action, name, n=n, pop=pop, increase=increase)

    def build_spawning_pool_hl(self, n, pop=0, increase=None, how=1, where='random'):
        action = lambda : self.build_spawning_pool(how=how, where=where)
        name = 'SpawningPool'
        self.main_hl(action, name, n=n, pop=pop, increase=increase)

    def move_workers_to_gas_hl(self, pop=0, increase=None, how='all'):
        action = lambda : self.move_workers_to_gas(how=how)
        self.main_hl(action, 'Hatchery', n=np.inf, pop=pop, increase=increase)

    def move_workers_to_minerals_hl(self, pop=0, increase=None, how='all'):
        action = lambda : self.move_workers_to_minerals(how=how)
        self.main_hl(action, 'Hatchery', n=np.inf, pop=pop, increase=increase)

    def wait_hl(self):
        if self.action is None:
            self.action = self.wait()


class ZerglingRush(ZergAgent):
    """Build : https://lotv.spawningtool.com/build/118355/"""

    def __init__(self):
        super().__init__()

    def step(self, obs):
        super().step(obs)

        # Build order
        # self.build_hatchery_hl(2, pop=14)
        # self.build_extractor_hl(1, pop=14)
        # self.build_spawning_pool_hl(1, pop=14)
        # self.train_overlord_hl(2, pop=14)

        self.build_extractor_hl(2, pop=16)
        if self.unit_count_by_type(units.Zerg.BroodLord) == 0:
            self.move_workers_to_gas_hl()
        else:
            self.move_workers_to_minerals_hl()
        self.train_brood_lord_hl(10, pop=22)
        self.train_zergling_hl(30, pop=22)
        self.wait_hl()

        return self.action


        ## Todo
        # Inject larvae
        # Rally point second base on minerals (idle workers)
        # Set rally point of bases when attack is launched
        # Micro

        # pysc2 missing microbial shroud?
