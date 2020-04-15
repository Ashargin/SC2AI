import itertools
from pysc2.lib.named_array import NamedNumpyArray
from pysc2.lib import features, actions, units, buffs
from settings import STEP_MUL


class Memory:
    def __init__(self, agent):
        self.agent = agent
        self.obs = None
        self.self_units = {}
        self.enemy_units = {}
        self.neutral_units = {}
        self.birth_dates = {}
        self.base_locations = []
        self.discarded_units = []
        self.has_rally_point = []
        self.spell_targets = {}
        self.expired_tumors = set()
        self.scouts = {}
        self.scout_timeout = {}
        self.creep_tumor_tries = {}
        self.function_timeout = {}
        self.game_step = 0

    def update(self, obs):
        self.obs = obs
        self.update_units()
        self.update_miscellaneous()
        self.game_step += 1

    def update_miscellaneous(self):
        try:
            for u in self.self_units[units.Zerg.CreepTumorBurrowed]:
                if u.order_id_0 == actions.RAW_FUNCTIONS.Build_CreepTumor_Tumor_pt.id:
                    self.expired_tumors.add(u.tag)
        except KeyError:
            pass

        def tick_timeout(timeout_dict):
            to_remove = []
            for key in timeout_dict:
                timeout_dict[key] -= STEP_MUL
                if timeout_dict[key] < 0:
                    to_remove.append(key)
            for key in to_remove:
                timeout_dict.pop(key)

        tick_timeout(self.scout_timeout)
        tick_timeout(self.function_timeout)

        for b in self.agent.get_units_agg('Hatchery'):
            if b.buff_id_0 == buffs.Buffs.QueenSpawnLarvaTimer:
                queen_tag = [q for q, h in self.spell_targets.items() if h == b.tag]
                if queen_tag:
                    self.spell_targets.pop(queen_tag[0])

    def update_units(self):
        alliances = self.obs.observation.raw_units[:, features.FeatureUnit.alliance]
        self_units = self.obs.observation.raw_units[alliances == features.PlayerRelative.SELF]
        enemy_units = self.obs.observation.raw_units[alliances == features.PlayerRelative.ENEMY]
        neutral_units = self.obs.observation.raw_units[alliances == features.PlayerRelative.NEUTRAL]

        killed = self.obs.observation._response_observation().observation.raw_data.event.dead_units
        self._update_units(self.self_units, self_units, killed)
        self._update_units(self.enemy_units, enemy_units, killed)
        self._update_units(self.neutral_units, neutral_units, killed)

    def _update_units(self, old, new, killed):
        # Only keep visible units
        displays = new[:, features.FeatureUnit.display_type]
        new = new[displays != 2] # remove snapshots
        new = [self._add_unit_variables(u) for u in new]

        # Add empty lists for previously unseen unit types
        visible_types = set([u.unit_type for u in new])
        for v_type in visible_types:
            if v_type not in old:
                old[v_type] = []

        # Update / remove old, unseen units
        new_tags = [u.tag for u in new]
        to_remove = [u.tag for u in new]
        missing = [u for u in list(itertools.chain.from_iterable(old.values())) 
                   if u.tag not in new_tags]
        zerg_build_fcts = []
        for unit in dir(units.Zerg):
            try:
                zerg_build_fcts.append(getattr(actions.RAW_FUNCTIONS, f'Build_{unit}_pt').id)
            except KeyError:
                pass
            try:
                zerg_build_fcts.append(getattr(actions.RAW_FUNCTIONS, f'Build_{unit}_unit').id)
            except KeyError:
                pass

        visibility_map = self.obs.observation.feature_minimap.visibility_map.T
        for u in missing:
            # Untrack dead units
            # Make sure that we are not cheating :
            # Only include units that we were seeing, or that are ours :
            died = u.tag in killed \
                   and (u.display_type != 2 or u.alliance == features.PlayerRelative.SELF)

            # Also untrack our zerg workers that are currently building
            died = died or (u.alliance == features.PlayerRelative.SELF
                           and u.unit_type == units.Zerg.Drone
                           and u.order_id_0 in zerg_build_fcts)

            if died or u.tag < 0:
                to_remove.append(u.tag)
                self.discarded_units.append(u.tag)
                for key in self.scouts:
                    if u.tag in self.scouts[key]:
                        self.scouts[key].remove(u.tag)
                if u.alliance == features.PlayerRelative.SELF \
                                        and u.unit_type in [units.Zerg.Hatchery,
                                                            units.Zerg.Lair,
                                                            units.Zerg.Hive,
                                                            units.Protoss.Nexus,
                                                            units.Terran.CommandCenter,
                                                            units.Terran.CommandCenterFlying,
                                                            units.Terran.OrbitalCommand,
                                                            units.Terran.OrbitalCommandFlying,
                                                            units.Terran.PlanetaryFortress]:
                    self.base_locations.remove((u.x, u.y))
                if u.alliance == features.PlayerRelative.SELF and u.unit_type == units.Zerg.Queen:
                    try:
                        self.spell_targets.pop(u.tag)
                    except KeyError:
                        pass
            else:
                u.display_type = 2 # seen
                u.time_unseen += 1
                u.time_alive += 1
                if visibility_map[u.x, u.y] == features.Visibility.VISIBLE:
                    u.pos_tracked = 0

        for key, type_units in old.items():
            old[key] = [u for u in type_units if u.tag not in to_remove]

        # Add / update new units
        for u in new:
            old[u.unit_type].append(u)

    def _add_unit_variables(self, unit):
        if unit.tag == 0:
            unit.tag = min(0, min(self.birth_dates.keys())) - 1
        if unit.tag not in self.birth_dates:
            self.birth_dates[unit.tag] = self.game_step
            if unit.unit_type in [units.Terran.CommandCenter, units.Protoss.Nexus, units.Zerg.Hatchery] \
                                                    and unit.alliance == features.PlayerRelative.SELF:
                self.base_locations.append((unit.x, unit.y))

        values = {'time_unseen': 0,
                  'time_alive': self.game_step - self.birth_dates[unit.tag],
                  'pos_tracked': 1}

        new_values = unit.tolist() + list(values.values())
        new_vars = list(unit._index_names[0].keys()) + list(values.keys())
        return NamedNumpyArray(new_values, new_vars)
