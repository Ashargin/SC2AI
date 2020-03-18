import os
from pysc2.env import sc2_env
from pysc2.lib import features
from absl import app
from agents import ZerglingRush

# Settings
agent = ZerglingRush()
op_race = 'random'
op_difficulty = 'easy' # very_easy, easy, medium, medium_hard, harder, very_hard
map_name = 'AbyssalReef'
visualize = True
realtime = False
save_replay_episodes = 1 # whether to save a replay

def main(unused_args):
    try:
        while True:
            with sc2_env.SC2Env(
                map_name=map_name,
                players=[sc2_env.Agent(getattr(sc2_env.Race, agent.race_name)),
                        sc2_env.Bot(getattr(sc2_env.Race, op_race),
                                    getattr(sc2_env.Difficulty, op_difficulty))],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True),
                step_mul=16,
                visualize=visualize,
                realtime=realtime,
                save_replay_episodes=save_replay_episodes,
                replay_dir='{}\\data\\replays'.format(os.path.abspath(os.getcwd()))) as env:

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

    except Exception as e:
        agent.log.close()
        raise Warning(e)

if __name__ == "__main__":
    app.run(main)


## Issues
# The AI may not find and destroy all enemy buildings even though it has defeated the opponent
# Clicks but repeatedly gets overlord (move them?)
# Pb build_queue / production queue : only for selection
# Why do I need to clip x, y?
# feature_units only returns units on the screen
