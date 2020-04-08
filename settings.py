from pysc2.lib import units

RESOLUTION = 200
THRESHOLD = int(RESOLUTION * 0.1)
STEP_MUL = 16
CAP_MINERALS = 400
CAP_GAS = 300

ATTACK_PRIORITY_GROUND = [units.Zerg.Queen,
                          units.Zerg.Ravager,
                          units.Zerg.Roach,
                          units.Zerg.Zergling,
                          units.Zerg.Drone,
                          units.Zerg.RoachWarren,
                          units.Zerg.SpawningPool,
                          units.Zerg.Hatchery,

                          units.Protoss.Zealot,
                          units.Protoss.Stalker,
                          units.Protoss.Adept,
                          units.Protoss.Probe,
                          units.Protoss.Pylon,
                          units.Protoss.Stargate,
                          units.Protoss.CyberneticsCore,
                          units.Protoss.Nexus,

                          units.Terran.SiegeTankSieged,
                          units.Terran.Marauder,
                          units.Terran.Reaper,
                          units.Terran.Marine,
                          units.Terran.SiegeTank,
                          units.Terran.SupplyDepot,
                          units.Terran.SupplyDepotLowered,
                          units.Terran.Barracks,
                          units.Terran.Factory,
                          units.Terran.Starport,
                          units.Terran.CommandCenter]
ATTACK_PRIORITY_AIR = []
ATTACK_PRIORITY_ALL = ATTACK_PRIORITY_GROUND + ATTACK_PRIORITY_AIR
ATTACK_PRIORITY_GROUND = {key: len(ATTACK_PRIORITY_GROUND) - i
                          for i, key in enumerate(ATTACK_PRIORITY_GROUND)}
ATTACK_PRIORITY_AIR = {key: len(ATTACK_PRIORITY_AIR) - i
                       for i, key in enumerate(ATTACK_PRIORITY_AIR)}
ATTACK_PRIORITY_ALL = {key: len(ATTACK_PRIORITY_ALL) - i
                       for i, key in enumerate(ATTACK_PRIORITY_ALL)}
ATTACK_PRIORITY = {'Ground': ATTACK_PRIORITY_GROUND,
                   'Air': ATTACK_PRIORITY_AIR,
                   'All': ATTACK_PRIORITY_ALL}


