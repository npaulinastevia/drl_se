import csv
import os

import gym
from gym.spaces import Box, Discrete
from skimage.draw import random_shapes

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from tensorboardX import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results,ts2xy

from wuji.problem.mdp.netease.blockmaze.maze import BaseMaze, Object, DeepMindColor as color, BaseEnv, \
    VonNeumannMotion
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,InputLayer
from tensorflow.keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint



class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf


    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')

            y=y.reshape((len(y),1))
            header = ['reward/episode']
            file = open('D:\drl_wuji\DQN_SB\A2C_SB_Training_rewards_1.csv', 'a+', newline='')
            with file:
                write = csv.writer(file)
                write.writerows(y)
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    #writer.add_scalar('DQN_SB_training_reward/episode', y, x)
                    #print(f"Num timesteps: {self.num_timesteps}")
                    print()
                    #print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    pass
                    #self.best_mean_reward = mean_reward
                    # Example for saving best model
                    #if self.verbose > 0:
                     #   print(f"Saving new best model to {self.save_path}")
                    #self.model.save(self.save_path)

        return True

def get_maze():
    size = (20, 20)
    max_shapes = 50
    min_shapes = max_shapes // 2
    max_size = 3
    seed = 2
    x, _ = random_shapes(size, max_shapes, min_shapes, max_size=max_size, multichannel=False, random_seed=seed)

    x[x == 255] = 0
    x[np.nonzero(x)] = 1

    # wall
    x[0, :] = 1
    x[-1, :] = 1
    x[:, 0] = 1
    x[:, -1] = 1

    return x


map = get_maze()
start_idx = [[10, 7]]
goal_idx = [[12, 12]]


class Maze(BaseMaze):
    @property
    def size(self):
        return map.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(map == 0), axis=1))
        obstacle = Object('obstacle', 85, color.obstacle, True, np.stack(np.where(map == 1), axis=1))
        agent = Object('agent', 170, color.agent, False, [])
        goal = Object('goal', 255, color.goal, False, [])
        return free, obstacle, agent, goal


class Env(BaseEnv):
    def __init__(self):
        super().__init__()

        self.maze = Maze()

        self.motions = VonNeumannMotion()
        # self.bugs = [
        #     [1,1],[3,4],[7,5],[18,1],[11,12],[18,14],
        #     [12,6],[18,6],[11,14],[1,13],[3,13],[1,17],
        #     [2,18],[10,18],[17,18],[12,18],[15,17]
        # ]
        # self.bugs = np.logical_and(np.random.randint(0,2,[20,20]), np.logical_not(map))
        # self.bugs_cnt = np.count_nonzero(self.bugs)
        self.bug_idxs = [[0, 1], [3, 4], [1, 6], [7, 5], [6, 17], [5, 11], [7, 1], [0, 10], [16, 10], [18, 1], [4, 1],
                         [11, 12], [18, 14], [12, 6], [18, 6], [11, 14], [1, 13], [3, 13], [1, 17], [2, 18], [10, 18],
                         [15, 3], [17, 18], [12, 18], [15, 17]]
        # self.bug_idxs = [[1, 2], [1, 6], [1, 7], [1, 8], [1, 17], [2, 2], [2, 3], [2, 7], [2, 9], [2, 10], [2, 11],
        #                  [2, 12], [2, 18], [3, 1], [3, 3], [3, 7], [3, 8], [3, 9], [3, 11], [3, 14], [3, 15], [3, 16],
        #                  [3, 17], [3, 18], [4, 1], [4, 2], [4, 3], [4, 7], [4, 9], [4, 11], [4, 14], [4, 17], [5, 4],
        #                  [5, 5], [5, 8], [5, 11], [5, 16], [6, 5], [6, 8], [7, 1], [7, 3], [7, 4], [7, 7], [7, 8],
        #                  [7, 9], [7, 11], [7, 17], [7, 18], [8, 1], [8, 2], [8, 8], [8, 9], [8, 11], [8, 12], [8, 13],
        #                  [8, 18], [9, 2], [9, 10], [9, 11], [9, 13], [9, 14], [10, 2], [10, 4], [10, 9], [10, 15],
        #                  [10, 16], [11, 10], [11, 11], [12, 1], [12, 4], [12, 5], [12, 6], [12, 11], [12, 12], [12, 13],
        #                  [13, 3], [13, 5], [13, 6], [13, 10], [13, 11], [14, 4], [14, 5], [14, 6], [14, 8], [14, 9],
        #                  [14, 10], [14, 16], [14, 17], [14, 18], [15, 1], [15, 2], [15, 5], [15, 6], [15, 7], [15, 8],
        #                  [15, 9], [15, 15], [16, 10], [17, 1], [17, 5], [17, 6], [17, 7], [17, 8], [17, 14], [17, 15],
        #                  [17, 17], [17, 18], [18, 1], [18, 2], [18, 3], [18, 4], [18, 6], [18, 7], [18, 12], [18,14]]
        self.bug_cnt = len(self.bug_idxs)

        #self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=(400,), dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

        self.context = dict(
            inputs=1,
            outputs=self.action_space.n
        )

    def step(self, action):
        motion = self.motions[action]

        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]

        # mark bug position
        bug = tuple(new_position) if new_position in self.bug_idxs else None

        # if bug is not None:
        #     print(bug)

        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        goal = self._is_goal(new_position)
        if goal:
            reward = +10
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False
        #return self.maze.to_value()[..., np.newaxis], reward, done, dict(bug=bug, valid=valid, goal=goal)

        return self.maze.to_value().reshape(-1), reward, done, dict(bug=bug, valid=valid, goal=goal, current=current_position, render=self.maze.to_value()[..., np.newaxis])

    def reset(self):
        self.bug_item = set()
        self.maze.objects.agent.positions = start_idx
        self.maze.objects.goal.positions = goal_idx

        #return self.maze.to_value()[..., np.newaxis], {}

        return self.maze.to_value().reshape(-1)

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out
    def get_visited_state(self, action):
        motion = self.motions[action]

        current_position = self.maze.objects.agent.positions[0]

        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]

        return new_position
    def get_image(self):
        return self.maze.to_rgb()


def sb_learn(algo='dqn'):

        from stable_baselines3 import A2C, DQN, PPO, DDPG
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        env = Env()
        lr = 0.00025
        policy_kwargs = dict(
            net_arch=[256, 128, 128,4]
        )
        if algo=='dqn':
            model = DQN("MlpPolicy", env, learning_rate=0.00025,
                    policy_kwargs=policy_kwargs, gamma=0.99, verbose=1, batch_size=128,exploration_initial_eps=1)
        if algo == 'ppo':
            model = PPO("MlpPolicy", env, learning_rate=0.00025, n_steps=128,
                        policy_kwargs=policy_kwargs, gamma=0.99, verbose=1)
        if algo == 'a2c':
            model = A2C("MlpPolicy", env, learning_rate=0.00025, n_steps=128, use_rms_prop=False,
                        policy_kwargs=policy_kwargs, gamma=0.99, rms_prop_eps=1e-08)
        model.learn(total_timesteps=10000,reset_num_timesteps=True)

        from stable_baselines3.common.distributions import CategoricalDistribution

        for i in range(10):
            numberOfbugs = 0
            buglist = []
            path = r'C:\Users\phili\Documents\Paulina Old\drl_wuji\DQN_SB_countVisitedstate' + str(i + 1)
            os.makedirs(path,exist_ok=True)
            writer = SummaryWriter(path)
            reward_sum=0
            obs = env.reset()
            n_iter=0
            prob_action=[0 for _ in range(4)]
            visited_state=[]
            while True:
                n_iter+=1
                action, _states = model.predict(obs)
                if action==0:
                    prob_action[0]+=1
                elif action==1:
                    prob_action[1]+=1
                elif action==2:
                    prob_action[2]+=1
                else:
                    prob_action[3]+=1
                temp=obs

                obs, rewards, dones, info = env.step(action)

                if info['current'] not in visited_state:
                    visited_state.append(info['current'])
                reward_sum+=rewards
                writer.add_scalar('agent_SB_reward_1/step', rewards, n_iter)
                writer.add_scalar('agent_SB_average_reward_1/steps', reward_sum/n_iter, n_iter)
                writer.add_scalar('agent_SB_cum_reward_1/steps', reward_sum, n_iter)
                print(numberOfbugs)
                if info['bug']:
                    if info['bug'] not in buglist:
                        numberOfbugs+=1
                        buglist.append(info['bug'])

                    #if n_iter%100==0:

                data= [n_iter,numberOfbugs]

                writer.add_scalar('PPO_SB_Bugs_1/Steps', numberOfbugs, n_iter)
                if n_iter==400000:
                    print(len(visited_state))
                    prob_action=[x/n_iter for x in prob_action]
                    file = open(f'{path}\PPO_SB_action_dist.csv', 'a+', newline='')
                    with file:
                        write = csv.writer(file)
                        write.writerow(prob_action) # each element is the probability of taking each action of the trained agent
                        write.writerow(visited_state) # each element is the total visited state of the maze
                        write.writerow([len(visited_state)])# this is the length of the visited state
                    obs_render=(info['render'],{})
                    plt.savefig('./maze.png')
                    plt.imshow(obs_render[0][..., 0])
                    for idx in env.bug_idxs:
                        obs_render[0][..., 0][idx[0], idx[1]] = 222
                    plt.imshow(obs_render[0][..., 0])
                    plt.savefig('./maze_with_bug.png')
                    break
                if dones:

                    obs = env.reset()

if __name__ == '__main__':
    algo='a2c'
    sb_learn(algo)




