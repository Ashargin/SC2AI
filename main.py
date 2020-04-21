import os
import cProfile
from pysc2.env import sc2_env
from pysc2.lib import features
from absl import app
from raw_agents import ZerglingRush, MacroZerg
from settings import RESOLUTION, STEP_MUL

# Settings
agent = MacroZerg()
op_race = 'terran' # random, terran, protoss, zerg
op_difficulty = 'very_hard' # very_easy, easy, medium, medium_hard, hard, harder, very_hard
op_build = 'macro' # random, rush, timing, power, macro, air
map_name = 'WorldofSleepers'
visualize = True
realtime = False
save_replay_episodes = 1 # whether to save a replay
time = False
time_iter = 800

def main(unused_args):
    try:
        while True:
            with sc2_env.SC2Env(
                map_name=map_name,
                players=[sc2_env.Agent(getattr(sc2_env.Race, agent.race_name)),
                         sc2_env.Bot(getattr(sc2_env.Race, op_race),
                                     getattr(sc2_env.Difficulty, op_difficulty),
                                     getattr(sc2_env.BotBuild, op_build))],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=RESOLUTION, minimap=RESOLUTION),
                    raw_resolution=RESOLUTION,
                    use_feature_units=True,
                    use_raw_units=True,
                    use_raw_actions=agent.raw_interface,
                    show_cloaked=True,
                    show_burrowed_shadows=True,
                    show_placeholders=True,
                    add_cargo_to_units=True,
                    crop_to_playable_area=True,
                    raw_crop_to_playable_area=True,
                    send_observation_proto=True
                    ),
                step_mul=STEP_MUL,
                visualize=visualize,
                realtime=realtime,
                save_replay_episodes=save_replay_episodes,
                replay_dir='{}\\data\\replays'.format(os.path.abspath(os.getcwd()))
                ) as env:

                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last() or time and agent.game_step > time_iter:
                        break
                    timesteps = env.step(step_actions)
                if time:
                    break
      
    except KeyboardInterrupt:
        agent.log.close()

if time:
    timed = cProfile.run('app.run(main)')
    print(timed)
elif __name__ == "__main__":
    app.run(main)
