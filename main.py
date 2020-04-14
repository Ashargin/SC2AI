import os
from pysc2.env import sc2_env
from pysc2.lib import features
from absl import app
from raw_agents import ZerglingRush, MacroZerg
from settings import RESOLUTION, STEP_MUL

# Settings
agent = ZerglingRush()
op_race = 'very_hard'
op_difficulty = 'medium' # very_easy, easy, medium, medium_hard, harder, very_hard
map_name = 'WorldofSleepers'
visualize = True
realtime = False
save_replay_episodes = 0 # whether to save a replay

def main(unused_args):
    try:
        while True:
            with sc2_env.SC2Env(
                map_name=map_name,
                players=[sc2_env.Agent(getattr(sc2_env.Race, agent.race_name)),
                         sc2_env.Bot(getattr(sc2_env.Race, op_race),
                                     getattr(sc2_env.Difficulty, op_difficulty))],
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
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
      
    except KeyboardInterrupt:
        agent.log.close()

if __name__ == "__main__":
    app.run(main)
