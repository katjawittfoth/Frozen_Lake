import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
import tensorflow as tf
from frozen_lake_env import FrozenLakeEnv
from IPython.display import clear_output


class DQNFrozenLakeAgent:

    def __init__(self):
        self.env = FrozenLakeEnv(map_name='4x4')

        self.states = np.identity(16)
        self.x = tf.placeholder(shape=[1, 16], dtype=tf.float32)

        self.W = tf.Variable(tf.random_uniform([16, 4], 0, 0.1))

        self.Q = tf.matmul(self.x, self.W)
        self.Q_hat = tf.placeholder(shape=[1, 4], dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.square(self.Q_hat - self.Q))
        self.train = tf.train.GradientDescentOptimizer(learning_rate=0.1)\
            .minimize(self.loss)

        self.gamma = 0.9
        self.epsilon = 1
        self.decay_rate = 0.001

        self.num_episodes = 500_000
        self.avg_rewards = []

        init = tf.initialize_all_variables()
        self.session = tf.Session()
        self.session.run(init)
        self.avg_rewards = []

    def test_matrix(self, Q, episode):
        total_reward = 0
        for i in range(100):
            state = self.env.reset()
            done = False
            while not done:
                Q_ = self.session.run(Q, feed_dict={self.x: self.states[state:state + 1]})
                a = np.argmax(Q_, 1)[0]

                state, r, done, _ = self.env.step(a)
                total_reward += r

        result = total_reward / 100
        print('Episode: {:,}, Average reward: {}'.format(episode, result))
        return result

    def epsilon_greedy(self, Q_pred):
        """
        Returns the next action by exploration with probability epsilon and
        exploitation with probability 1-epsilon.
        """
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(Q_pred, 1)[0]

    def decay_epsilon(self, episode):
        """
        Decaying exploration with the number of episodes.
        """
        self.epsilon = 0.1 + 0.9 * np.exp(-self.decay_rate * episode)

    def run_training(self):
        """Training the agent to find the frisbee on the frozen lake"""

        self.avg_rewards = []
        self.episode_len = np.zeros(self.num_episodes)

        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                # Predicted Q
                Q_pred = self.session.run(self.Q, {self.x: self.states[state:state + 1]})
                action = self.epsilon_greedy(Q_pred)
                new_state, reward, done, _ = self.env.step(action)

                # Actual Q after performing an action
                Q_true = reward + self.gamma * np.max(
                    self.session.run(self.Q, {self.x: self.states[new_state:new_state + 1]}))

                Q_pred[0, action] = Q_true

                # Calculate loss and train the agent
                self.session.run(self.train, feed_dict={self.x: self.states[state:state + 1], self.Q_hat: Q_pred})

                state = new_state

                self.episode_len[episode] += 1

            self.decay_epsilon(episode)

            if episode % 5000 == 0:
                avg_reward = self.test_matrix(self.Q, episode)
                self.avg_rewards.append(avg_reward)
                if avg_reward > 0.8:
                    print("Frozen Lake solved 🏆🏆🏆")
                    break

    def plot(self):
        """Plot the episode length and average rewards per episode"""

        fig = plt.figure(figsize=(20, 5))

        episode_len = [i for i in self.episode_len if i != 0]

        rolling_len = pd.DataFrame(episode_len).rolling(100, min_periods=100)
        mean_len = rolling_len.mean()
        std_len = rolling_len.std()

        plt.plot(mean_len, color='red')
        plt.fill_between(x=std_len.index, y1=(mean_len - std_len)[0],
                         y2=(mean_len + std_len)[0], color='red', alpha=.2)

        plt.ylabel('Episode length')
        plt.xlabel('Episode')
        plt.title(
            f'Frozen Lake - Length of episodes (mean over window size 100)')
        plt.show(fig)

        fig = plt.figure(figsize=(20, 5))

        plt.plot(self.avg_rewards, color='red')
        plt.gca().set_xticklabels(
            [i + i * 4999 for i in range(len(self.avg_rewards))])

        plt.ylabel('Average Reward')
        plt.xlabel('Episode')
        plt.title(f'Frozen Lake - Average rewards per episode ')
        plt.show(fig)


def play(agent, num_episodes=1):
    """Let the agent play Frozen Lake"""
    time.sleep(2)
    for episode in range(num_episodes):
        state = agent.env.reset()
        done = False
        print('❄️🕳🥶 Frozen Lake - Episode ', episode + 1,
              '⛸🥏🏆 \n\n\n\n\n\n\n\n')

        time.sleep(1.5)

        steps = 0
        while not done:
            clear_output(wait=True)
            agent.env.render()
            time.sleep(0.3)

            Q_ = agent.session.run(agent.Q, feed_dict={agent.x: agent.states[state:state + 1]})
            action = np.argmax(Q_, 1)[0]

            state, reward, done, _ = agent.env.step(action)
            steps += 1

        clear_output(wait=True)
        agent.env.render()

        if reward == 1:
            print(f'Yay! 🏆You have found your 🥏 in {steps} steps.')
            time.sleep(2)
        else:
            print('Oooops 🥶 you fell through a 🕳, try again!')
            time.sleep(2)
        clear_output(wait=True)


agent = DQNFrozenLakeAgent()
agent.run_training()
play(agent)
agent.plot()
