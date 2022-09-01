import csv
import os

import gym
from gym.spaces import Box, Discrete
from skimage.draw import random_shapes
import numpy as np
import matplotlib.pyplot as plt

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

        self.observation_space = Box(low=0, high=255, shape=(400,), dtype=np.uint8)
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
            done = 1
        elif not valid:
            reward = -1
            done = 0
        else:
            reward = -0.01
            done = 0
        #return self.maze.to_value()[..., np.newaxis], reward, done, dict(bug=bug, valid=valid, goal=goal)

        return self.maze.to_value().reshape(-1), reward, done, dict(bug=bug, valid=valid, goal=goal,current=current_position, render=self.maze.to_value()[..., np.newaxis])

    def execute(self, action):

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

        return self.maze.to_value().reshape(-1), reward, done, dict(bug=bug, valid=valid, goal=goal)


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

    def get_image(self):
        return self.maze.to_rgb()


def aaa():
    env.reset()
    cnt = 0
    while True:
        s, r, d, info = env.step(np.random.randint(0, 4))
        print(info)
        cnt += 1
        if cnt >= 100:
            break


if __name__ == '__main__':

    NAME = os.path.basename(os.path.dirname(os.path.splitext(__file__)[0]))
    LENGTH = 40
    env = Env()
    lr = 0.00025
    algo = 'ppo'
    from tensorforce.environments import Environment
    from tensorforce.agents import Agent
    from tensorboardX import SummaryWriter
    for i in range(10):
          step=0
          environment = Environment.create(
              environment=env, max_episode_timesteps=40
          )
          if algo=='ppo':
              agent = Agent.create(
                  agent='ppo', environment=environment, parallel_interactions=4, network=[
                      dict(type='flatten'),
                      dict(type='dense', size=256, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='linear', size=4),
                  ],
                  discount=0.99, memory=10000, learning_rate=0.00025, batch_size=128,

              )
          if algo == 'dqn':
              agent = Agent.create(
                  agent='dqn', environment=environment, network=[
                      dict(type='flatten'),
                      dict(type='dense', size=256, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='linear', size=4),
                  ],
                  discount=0.99, memory=10000, learning_rate=0.00025, batch_size=128
              )
          if algo == 'a2c':
              agent = Agent.create(
                  agent='a2c', environment=environment, parallel_interactions=4, network=[
                      dict(type='flatten'),
                      dict(type='dense', size=256, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='linear', size=4),
                  ],
                  discount=0.99, memory=10000, learning_rate=0.00025, batch_size=128
              )

          for j in range(128):
              states = environment.reset()
              terminal = 0
              if step == 10000:
                  break
              while terminal==0:

                  if step == 10000:
                      break
                  actions = agent.act(states=states)

                  states, terminal,reward = environment.execute(actions=actions)

                  agent.observe(terminal=terminal, reward=reward)
                  step =step + 1


          numberOfbugs=0
          obs = environment.reset()
          path= r'C:\Users\phili\Documents\Paulina Old\drl_wuji\DQN_SB_countVisitedstate' + str(i + 1)
          os.makedirs(path,exist_ok=True)
          writer1 = SummaryWriter(path)
          n_iter=0
          buglist = []
          visited_state=[]
          reward_sum=0
          n_iter=0
          prob_action=[0 for _ in range(4)]
          while True:
              n_iter+=1
              action= agent.act(states=obs)

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
              writer1.add_scalar('agent_tensor_reward/step', rewards, n_iter)
              writer1.add_scalar('agent_tensor_average_reward/steps', reward_sum/n_iter, n_iter)
              writer1.add_scalar('agent_tensor_cum_reward/steps', reward_sum, n_iter)
              agent.observe(terminal=terminal, reward=reward)
              print(numberOfbugs)
              if info['bug']:
                  if info['bug'] not in buglist:
                      numberOfbugs+=1
                      buglist.append(info['bug'])

              writer1.add_scalar('(agent)_Tensor_Bugs/Steps', numberOfbugs, n_iter)
              if n_iter==4000000:
                  prob_action=[x/n_iter for x in prob_action]
                  file = open(f'{path}\agent_TF_action_dist.csv', 'a+', newline='')
                  with file:
                      write = csv.writer(file)
                      write.writerow(prob_action)
                      write.writerow(visited_state)
                      write.writerow([len(visited_state)])
                  obs_render=(info['render'],{})
                  for idx in env.bug_idxs:
                      obs_render[0][..., 0][idx[0], idx[1]] = 222
                  plt.imshow(obs_render[0][..., 0])
                  plt.savefig('./maze_with_bug.png')
                  break
              if dones:
                  obs = env.reset()
