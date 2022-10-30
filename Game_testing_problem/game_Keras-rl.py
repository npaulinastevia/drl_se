import csv
import os

import gym
import tensorflow
from gym.spaces import Box, Discrete
from skimage.draw import random_shapes

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from tensorboardX import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results,ts2xy
from keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from keras.layers import Dense, Activation, Flatten, Input, Concatenate,InputLayer
from rl.agents.ddpg import DDPGAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from wuji.problem.mdp.netease.blockmaze.maze import BaseMaze, Object, DeepMindColor as color, BaseEnv, \
    VonNeumannMotion

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



def kr_learn(algo='dqn'):


        env = Env()

        lr = 0.00025
        policy_kwargs = dict(
            net_arch=[256, 128, 128,4]
        )
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                      value_max=1, value_min=0.05, value_test=.2, nb_steps=10000)
        memory = SequentialMemory(limit=10000, window_length=1)
        nb_actions = env.action_space.n
        if algo == 'dqn':
            model = Sequential()
            model.add(Flatten(input_shape=((1,)+env.observation_space.shape)))

            model.add(Dense(256, activation="relu"))
            model.add(Dense(128, activation="relu"))
            model.add(Dense(128, activation="relu"))
            model.add(Dense(nb_actions))
            model.add(Activation('linear'))

            dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
                       memory=memory,
                       gamma=.99, batch_size=128, enable_double_dqn=False)

            dqn.compile(Adam(lr=.00025), metrics=['mae'])
            dqn.fit(env, nb_steps=10000,visualize=False,verbose=1)
            agent=dqn

        for i in range(10):
            n_iter = 0
            episode_num = 0
            buglist = []
            numberOfbugs = 0
            path = r'C:\Users\phili\Documents\Paulina Old\drl_wuji\DQN_SB_countVisitedstate' + str(i+1)
            os.makedirs(path,exist_ok=True)
            writer = SummaryWriter(path)
            reward_sum=0
            obs = env.reset()
            n_iter=0
            prob_action=[0 for _ in range(4)]
            visited_state=[]
            while True:
                n_iter+=1
                #obs=obs.reshape(400,)
                action = agent.forward(obs)
                if action==0:
                    prob_action[0]+=1
                elif action==1:
                    prob_action[1]+=1
                elif action==2:
                    prob_action[2]+=1
                else:
                    prob_action[3]+=1
                obs, rewards, dones, info = env.step(action)
                if info['current'] not in visited_state:
                    visited_state.append(info['current'])
                reward_sum+=rewards
                writer.add_scalar('agent_KR_reward_1/step', rewards, n_iter)
                writer.add_scalar('agent_KR_average_reward_1/steps', reward_sum/n_iter, n_iter)
                writer.add_scalar('agent_KR_cum_reward_1/steps', reward_sum, n_iter)
                print(numberOfbugs)
                if info['bug']:
                    if info['bug'] not in buglist:
                        numberOfbugs+=1
                        buglist.append(info['bug'])

                    #if n_iter%100==0:

                data= [n_iter,numberOfbugs]
                writer.add_scalar('agent_KR_Bugs_1/Steps', numberOfbugs, n_iter)
                if n_iter==4000000:
                    prob_action=[x/n_iter for x in prob_action]
                    file = open(f'{path}\agent_KR_action_dist.csv', 'a+', newline='')
                    with file:
                        write = csv.writer(file)
                        write.writerow(prob_action) # each element is the probability of taking each action of the trained agent
                        write.writerow(visited_state) # each element is the total visited state of the maze
                        write.writerow([len(visited_state)])# this is the length of the visited state
                    obs_render=(info['render'],{})
                    plt.savefig('./mazeDQNKR.png')
                    plt.imshow(obs_render[0][..., 0])
                    for idx in env.bug_idxs:
                        obs_render[0][..., 0][idx[0], idx[1]] = 222
                    plt.imshow(obs_render[0][..., 0])
                    plt.savefig('./mazeDQNKR_with_bug.png')
                    break
                if dones:
                    obs = env.reset()


if __name__ == '__main__':
    from tensorflow.python.framework.ops import disable_eager_execution
    tensorflow.compat.v1.disable_eager_execution()
    algo='dqn'
    kr_learn(algo)




