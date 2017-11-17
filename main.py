import gym
import argparse
from NeuralNetwork import DeepQAgent, NeuralModel

ENV = 'Breakout-v0'
NETWORK_PATH = 'saved_networks/' + ENV


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default=ENV)
    parser.add_argument('--render', dest='render',
                        action='store_true', default=False)
    parser.add_argument('--evaluate', dest='evaluate',
                        action='store_true', default=False)

    args = parser.parse_args()

    env_name = args.env
    render = args.render
    evaluate = args.evaluate
    print("render", render)
    env = gym.make(env_name)

    brain = NeuralModel(gamma=0.99,
                        num_actions=env.action_space.n,
                        min_grad=0.01,
                        momentum=0.95,
                        learning_rate=2.5e-4,
                        weight_path=NETWORK_PATH)
    agent = DeepQAgent(
        brain=brain,
        num_episodes=2e5, env=env,
        max_steps=1000000,
        epsilon=1.0, render=render,
        min_epsilon=0.1,
        epsilon_decay_steps=1000000,
        replay_queue_size=1000000,
        sample_batch_size=32,
        warmup_steps=5e4,
        update_rate=4,
        target_update=10000,
        save_rate=200000
    )

    if evaluate:
        agent.evaluate()
    else:
        agent.train()


if __name__ == '__main__':
    main()





# import gym
# from scipy.misc import imsave
# from gym.wrappers import Monitor
# from utils import preprocess_frame
# from NeuralNetwork import DeepQAgent
#
# BATCH_SIZE = 16
# BUFFER_START_SIZE = 40000
# BUFFER_SIZE = 1000000
# TARGET_UPDATE = 50
# NUM_EPISODES = int(2e5)
# LEARNING_RATE = 2.5e-4
# MAX_STEPS = int(1e6)
# GAMMA = 0.99
# EPSILON_DECAY_STEPS = int(1e6)
# MIN_EPS = 0.1
#
#
# env = gym.make("Breakout-v0")
# # env = Monitor(env, directory='video/breakout')
#
# agent = DeepQAgent(replay_queue_size=BUFFER_SIZE, env=env, sample_batch_size=BATCH_SIZE, buffer_start_size=BUFFER_START_SIZE)
#
# env.reset()
#
# for a in range(NUM_EPISODES):
#
#
#
#     obs, r, done, _ = env.step(action)
#     obs = preprocess_frame(obs)
#
#     if done:
#         break
#
#     # imsave('out/'+str(a)+'-'+str(action)+'-'+str(r)+'.png', obs)
#
#
#
# # print(env.action_space)
# # print(env.observation_space)
# # print(env.action_space.sample())
# # print(env.action_space.high)
#
#
#
