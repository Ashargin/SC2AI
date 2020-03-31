import itertools
from pysc2.lib.named_array import NamedNumpyArray
from pysc2.lib import features, actions, units


class Memory:
    def __init__(self):
        self.self_units = {}
        self.op_units = {}
        self.neutral_units = {}
        self.birth_dates = {}
        self.base_locations = []
        self.game_step = 0

    def update(self, obs):
        self.update_units(obs)
        self.game_step += 1

    def update_units(self, obs):
        self_units = [u for u in obs.observation.raw_units
                      if u.alliance == features.PlayerRelative.SELF]
        op_units = [u for u in obs.observation.raw_units
                    if u.alliance == features.PlayerRelative.ENEMY]
        neutral_units = [u for u in obs.observation.raw_units
                         if u.alliance == features.PlayerRelative.NEUTRAL]

        killed = obs.observation._response_observation().observation.raw_data.event.dead_units
        self._update_units(self.self_units, self_units, killed)
        self._update_units(self.op_units, op_units, killed)
        self._update_units(self.neutral_units, neutral_units, killed)

    def _update_units(self, old, new, killed):
        # Only keep visible units
        new = [self._add_unit_variables(u) for u in new if u.display_type != 2] # remove snapshots

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
            else:
                u.display_type = 2 # seen
                u.time_unseen += 1
                u.time_alive += 1

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
            if unit.unit_type in [units.Terran.CommandCenter, units.Protoss.Nexus, units.Zerg.Hatchery]:
                self.base_locations.append((unit.x, unit.y))

        values = {'time_unseen': 0,
                  'time_alive': self.game_step - self.birth_dates[unit.tag]}

        new_values = unit.tolist() + list(values.values())
        new_vars = list(unit._index_names[0].keys()) + list(values.keys())
        return NamedNumpyArray(new_values, new_vars)
