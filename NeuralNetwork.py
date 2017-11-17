import os
import numpy as np
import tensorflow as tf
from keras.losses import MSE
from keras.models import Sequential
from keras import initializers
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import RMSprop
from keras import backend as K
from collections import deque
from gym.wrappers import Monitor

from utils import preprocess_frame, timeit, repeat_upsample
from gym.envs.classic_control import rendering 
from ReplayBuffer import ReplayBuffer


class NeuralModel:
    def __init__(self, gamma, min_grad, momentum, num_actions, learning_rate,
                 weight_path=None):
        self.num_actions = num_actions
        self.min_grad = min_grad
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.weight_path = weight_path

        # Online model
        self.online_input, self.online_q_vals, online_network = self.build_network()
        online_network_weights = online_network.trainable_weights

        # Target model
        self.target_input, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        self.update_target_network = [
            target_network_weights[i].assign(online_network_weights[i])
            for i in range(len(target_network_weights))]

        self.actions, self.targets, self.loss, self.grads_update = self.build_training_op(
            online_network_weights)

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(online_network_weights)

        self.sess.run(tf.global_variables_initializer())

        if weight_path:
            if not os.path.exists(weight_path):
                os.makedirs(weight_path)
            self.load_network()

        # Set same weights to both networks at start
        self.sess.run(self.update_target_network)

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(self.weight_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Loaded weights: " + checkpoint.model_checkpoint_path)
        else:
            print("failed loading weights.. training new network")

    def save_weight(self, global_step, mean_reward_100):
        save_path = self.saver.save(self.sess, self.weight_path + '/r' + str(
            int(mean_reward_100)), global_step=global_step)
        print("Saved weights:" + save_path)

    def build_training_op(self, online_network_weights):
        actions = tf.placeholder(tf.int64, [None])
        targets = tf.placeholder(tf.float32, [None])

        one_hot_actions = tf.one_hot(actions, self.num_actions, 1.0, 0.0)
        q_values = tf.reduce_sum(
            tf.multiply(self.online_q_vals, one_hot_actions),
            reduction_indices=1)

        error = tf.abs(targets - q_values)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate,
                                              momentum=self.momentum,
                                              epsilon=self.min_grad)
        grads_update = optimizer.minimize(loss, var_list=online_network_weights)

        return actions, targets, loss, grads_update

    # def update_weights(self):
    #     self.target_model.set_weights(self.model.get_weights())

    def update_weights(self):
        self.sess.run(self.update_target_network)

    def build_network(self):
        model = Sequential()
        model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu',
                                input_shape=(84, 84, 4)))
        model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions))

        input_placeholder = tf.placeholder(tf.float32, [None, 84, 84, 4])
        q_values = model(input_placeholder)

        return input_placeholder, q_values, model

    def train_model(self, mini_batch):
        obses, actions, rewards, obses_t, dones = mini_batch

        # Convert booleans to integers
        dones = dones + 0

        target_q_vals = self.target_q_values.eval(
            feed_dict={self.target_input: np.float32(np.array(obses_t))})
        targets = rewards + (1 - dones) * self.gamma * np.max(target_q_vals,
                                                              axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.online_input: np.float32(np.array(obses)),
            self.actions: actions,
            self.targets: targets
        })

    def choose_action(self, obs):
        obs1 = np.stack(obs, axis=2)
        obs1 = np.expand_dims(obs1, axis=0)

        # q_value = self.model.predict(a)
        action = np.argmax(
            self.online_q_vals.eval(feed_dict={self.online_input: obs1}))
        return action


class DeepQAgent:
    def __init__(self, brain, replay_queue_size, env, max_steps,
                 sample_batch_size,
                 epsilon_decay_steps, save_rate, min_epsilon,
                 num_episodes,
                 epsilon, render,
                 warmup_steps, update_rate, target_update):

        self.update_rate = update_rate
        self.target_update = target_update
        self.save_rate = save_rate
        self.history_size = 4

        self.replay_buffer = ReplayBuffer(replay_queue_size)

        self.render = render
        if render:
            # self.env = Monitor(env, directory='/video')
            self.env = env
        else:
            self.env = env
        self.num_actions = env.action_space.n

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps

        self.warmup = True
        self.warmup_steps = warmup_steps

        self.sample_batch_size = sample_batch_size
        self.num_episodes = int(num_episodes)
        self.max_steps = max_steps
        self.recent_observations = deque(maxlen=self.history_size)
        self.reward_log = deque(100 * [0], 100)
        self.render = render

        self.brain = brain

    def train(self):

        total_steps = 0
        for ep in range(self.num_episodes):
            steps = 0
            rewards = 0

            first_obs, dn = self.new_random_game()
            self.recent_observations = deque(maxlen=self.history_size)
            self.recent_observations.append(preprocess_frame(first_obs))

            while steps < self.max_steps:

                action = self.select_action()
                if self.render:
                    self.env.render()
                second_obs, reward, done, _ = self.env.step(action)

                self.replay_buffer.append(self.recent_observations[-1], action,
                                          reward, done)
                self.append_to_episode_observations(second_obs)

                rewards += reward
                steps += 1
                total_steps += 1

                if total_steps > self.warmup_steps:
                    self.warmup = False

                if not self.warmup:
                    if total_steps % self.update_rate == 0:
                        self.train_model()
                    if total_steps % (
                        self.target_update * self.update_rate) == 0:
                        self.brain.update_weights()
                    if self.epsilon > self.min_epsilon:
                        self.epsilon -= (1.0 - self.min_epsilon) / \
                                        self.epsilon_decay_steps
                    if total_steps % self.save_rate == 0:
                        self.brain.save_weight(total_steps,
                                               np.mean(self.reward_log))

                if done:
                    self.report(total_steps, steps, rewards, ep, self.epsilon)
                    break

    def evaluate(self):
        total_steps = 0
        viewer = rendering.SimpleImageViewer()
        for ep in range(self.num_episodes):
            steps = 0
            rewards = 0

            first_obs = self.env.reset()
            self.recent_observations = deque(maxlen=self.history_size)
            self.recent_observations.append(preprocess_frame(first_obs))

            while steps < 1000000:
                
                if np.random.rand() < 0.05:
                    action = 1
                else:
                    action = self.select_test_action()
                
                if self.render:
                    rgb = self.env.render('rgb_array')
                    upscaled = repeat_upsample(rgb, 3, 3)
                    viewer.imshow(upscaled)
                    #self.env.render()
                second_obs, reward, done, _ = self.env.step(action)
                self.append_to_episode_observations(second_obs)
                rewards += reward
                steps += 1
                total_steps += 1
                print(steps, " ", reward)
                if done:
                    self.report(total_steps, steps, rewards, ep, self.epsilon)
                    break


    def train_model(self):
        mini_batch = self.replay_buffer.sample_random(self.sample_batch_size,
                                                      self.history_size)
        self.brain.train_model(mini_batch)

    def new_random_game(self):
        ob = self.env.reset()
        done = False
        no_rnd = np.random.randint(0, 30)
        for i in range(no_rnd):
            ob, _, done, _ = self.env.step(0)
        return ob, done

    def append_to_episode_observations(self, obs):
        preproc_obs = preprocess_frame(obs)
        self.recent_observations.append(preproc_obs)

    def select_action(self):
        rnd = np.random.uniform(0, 1)
        if self.warmup or rnd < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            obs = list(self.recent_observations)
            while len(obs) < 4:
                obs = [obs[0]] + obs
            action = self.brain.choose_action(obs)
        return action

    def select_test_action(self):
        obs = list(self.recent_observations)
        while len(obs) < 4:
            obs = [obs[0]] + obs
        action = self.brain.choose_action(obs)
        return action

    def report(self, total_steps, steps, rewards, episode, epsilon):
        self.reward_log.append(rewards)
        print('Episode: {} Total steps: {}, steps: {}, reward: {} mean-100: '
              '{} epsilon: {}'.format(episode, total_steps, steps, rewards,
                                      np.mean(self.reward_log), epsilon))
