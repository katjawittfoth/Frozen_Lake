import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from frozen_lake_env import FrozenLakeEnv
from IPython.display import clear_output


class QFrozenLakeAgent:

    def __init__(self, num_episodes=500_000, max_steps=200, learning_rate=0.01, gamma=0.9, epsilon=1.0, decay_rate=0.01):
        self.env = FrozenLakeEnv()
        state_size = self.env.observation_space.n  # 4x4 ==> 16
        actions_num = self.env.action_space.n  # four actions {LEFT: 0, DOWN: 1, RIGHT: 2, UP
        self.q_table = np.zeros((state_size, actions_num))
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.avg_rewards = []

    def test_matrix(self, q_table, episode):
        total_reward = 0
        for i in range(100):
            s = self.env.reset()
            done = False
            while not done:
                a = np.argmax(q_table[s])
                s, r, done, _ = self.env.step(a)
                total_reward += r

        result = total_reward / 100
        print('Episode: {:,}, Average reward: {}'.format(episode, result))
        return result

    def update_q_table(self, state, action):
        """
        Using Bellman equation updates Q Table with action, state and it's reward.
        """
        new_state, reward, done, _ = self.env.step(action)

        self.q_table[state, action] = self.q_table[
                                          state, action] + self.learning_rate * (
                                                  reward + self.gamma * np.max(
                                              self.q_table[new_state]) -
                                                  self.q_table[state, action])
        return new_state, reward, done

    def epsilon_greedy(self, state):
        """
        Returns the next action by exploration with probability epsilon and exploitation with probability 1-epsilon.
        """
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def decay_epsilon(self, episode):
        """
        Decaying exploration with the number of episodes.
        """
        self.epsilon = 0.1 + 0.9 * np.exp(-self.decay_rate * episode)

    def train(self):
        """Training the agent to find the frisbee on the frozen lake"""

        self.avg_rewards = []
        self.episode_len = np.zeros(self.num_episodes)

        for episode in range(self.num_episodes):
            state = self.env.reset()

            for step in range(self.max_steps):
                action = self.epsilon_greedy(state)
                state, reward, done = self.update_q_table(state, action)

                self.episode_len[episode] += 1

                if done:
                    break

            self.decay_epsilon(episode)

            if episode % 1000 == 0:
                avg_reward = self.test_matrix(self.q_table, episode)
                self.avg_rewards.append(avg_reward)
                if avg_reward > 0.8:
                    # considered "solved" when the agent get an avg of at least 0.78 over 100 in a row.
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
            [i + i * 999 for i in range(len(self.avg_rewards))])

        plt.ylabel('Average Reward')
        plt.xlabel('Episode')
        plt.title(f'Frozen Lake - Average rewards per episode ')
        plt.show(fig)


def play(agent, num_episodes=5):
    """Let the agent play Frozen Lake"""

    time.sleep(2)

    for episode in range(num_episodes):
        state = agent.env.reset()
        done = False
        print('❄️🕳🥶 Frozen Lake - Episode ', episode + 1, '⛸🥏🏆 \n\n\n\n')

        time.sleep(1.5)

        steps = 0
        while not done:
            clear_output(wait=True)
            agent.env.render()
            time.sleep(0.3)

            action = np.argmax(agent.q_table[state])
            state, reward, done, _ = agent.env.step(action)
            steps += 1

        clear_output(wait=True)
        agent.env.render()

        if reward == 1:
            print(f'Yay! 🏆 You have found your 🥏 in {steps} steps.')
            time.sleep(2)
        else:
            print('Oooops 🥶 you fell through a 🕳, try again!')
            time.sleep(2)
        clear_output(wait=True)


# agent = QFrozenLakeAgent()
# agent.train()
# play(agent, num_episodes=1)
# agent.plot()
