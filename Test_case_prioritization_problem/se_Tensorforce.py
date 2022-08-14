import argparse

import numpy
import pandas as pd
import numpy as np
import math
import os
from datetime import datetime
from statistics import mean
import tensorflow as tf
import random
from keras.callbacks_v1 import TensorBoard
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate,InputLayer
from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from TPAgentUtil import TPAgentUtil
from PairWiseEnv import CIPairWiseEnv
from ci_cycle import CICycleLog
from Config import Config
from TestcaseExecutionDataLoader import TestCaseExecutionDataLoader
from pathlib import Path
from CIListWiseEnv import CIListWiseEnv
from PointWiseEnv import CIPointWiseEnv
import sys


def test_agent(env: CIPairWiseEnv, model_path: str, algo, mode):
    agent_actions = []

    print("Evaluation of an agent from " + model_path)

    # model = TPAgentUtil.load_model(path=model_path, algo=algo, env=env)
    from tensorforce.environments import Environment
    from tensorforce.agents import Agent
    environment = Environment.create(
        environment=env
    )
    model = Agent.load(directory='model-numpy', format='numpy', environment=environment)
    if model:

        if mode.upper() == "PAIRWISE" and algo.upper() != "DQN":
            obs = env.reset()
            states = env.reset()
            done = 0
            step_t = 0
            test_rew = 0
            while done == 0 and step_t < 300:
                step_t += 1
                actions = model.act(states=states)

                obs, rewards, done, info = env.step(actions)
                test_rew += rewards
                model.observe(terminal=done, reward=rewards)

                if done:
                    break
            return env.sorted_test_cases_vector, test_rew
        elif mode.upper() == "PAIRWISE" and algo.upper() == "DQN":
            obs = env.reset()
            #
            done = False
            test_rew = 0
            while True:
                action = model.act(states=obs)
                obs, rewards, done, info = env.step(action)
                model.observe(terminal=done, reward=rewards)
                test_rew += rewards
                if done:
                    break
            return env.sorted_test_cases_vector, test_rew
        elif mode.upper() == "POINTWISE":

            if model:
                test_cases = env.cycle_logs.test_cases
                obs = env.reset()
                done = False
                index = 0
                test_cases_vector_prob = []
                test_rew = 0
                for index in range(0, len(test_cases)):
                    # action= model.forward(obs)
                    action = model.act(states=obs)
                    obs, rewards, done, info = env.step(action)
                    test_rew += rewards
                    model.observe(terminal=done, reward=rewards)
                    test_cases_vector_prob.append({'index': index, 'prob': action})
                    if done:
                        assert len(test_cases) == index + 1, "Evaluation is finished without iterating all " \
                                                             "test cases "
                        break
                test_cases_vector_prob = sorted(test_cases_vector_prob, key=lambda x: x['prob'],
                                                reverse=False)  ## the lower the rank, te higher the priority
                sorted_test_cases = []
                for test_case in test_cases_vector_prob:
                    sorted_test_cases.append(test_cases[test_case['index']])
            return sorted_test_cases, test_rew
            pass
        elif mode.upper() == "LISTWISE":
            if model:
                test_cases = env.cycle_logs.test_cases
                obs = env.reset()
                done = False
            i = 0
            test_rew = 0
            while True and i < 1000000:
                i = i + 1
                action = model.act(states=obs)
                if agent_actions.count(action) == 0 and action < len(test_cases):
                    if isinstance(action, list) or isinstance(action, np.ndarray):
                        agent_actions.append(action[0])
                    else:
                        agent_actions.append(action)

                obs, rewards, done, info = env.step(action)
                model.observe(terminal=done, reward=rewards)
                test_rew += rewards
                if done:
                    break
            sorted_test_cases = []

            for index in agent_actions:
                sorted_test_cases.append(test_cases[index])
            if (i >= 1000000):
                sorted_test_cases = test_cases
            return sorted_test_cases, test_rew


def millis_interval(start, end):
    """start and end are datetime instances"""
    diff = end - start
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis


## find the cycle with maximum number of test cases
def get_max_test_cases_count(cycle_logs:[]):
    max_test_cases_count = 0
    for cycle_log in cycle_logs:
        if cycle_log.get_test_cases_count()>max_test_cases_count:
            max_test_cases_count = cycle_log.get_test_cases_count()
    return max_test_cases_count


def experiment(mode, algo, test_case_data, start_cycle, end_cycle, episodes, model_path, dataset_name, conf,verbos=False):
    log_dir = os.path.dirname(conf.log_file)
#    -- fix end cycle issue
    total_steps = 0

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if start_cycle <= 0:
        start_cycle = 0

    if end_cycle >= len(test_case_data)-1:
        end_cycle = len(test_case_data)
    ## check for max cycle and end_cycle and set end_cycle to max if it is larger than max
    log_file = open(conf.log_file, "a")
    log_file_test_cases = open(log_dir+"/sorted_test_case.csv", "a")
    log_file.write("timestamp,mode,algo,model_name,episodes,steps,cycle_id,training_time,testing_time,winsize,test_cases,failed_test_cases, apfd, nrpa, random_apfd, optimal_apfd,accumulated_reward_test,accumulated_reward_train" + os.linesep)
    first_round: bool = True
    if start_cycle > 0:
        first_round = False
        previous_model_path = model_path + "/" + mode + "_" + algo + dataset_name + "_" + str(
            0) + "_" + str(start_cycle-1)
    model_save_path = None
    apfds=[]
    nrpas=[]
    for i in range(start_cycle, end_cycle - 1):
        if (test_case_data[i].get_test_cases_count() < 6) or \
                ( (conf.dataset_type == "simple") and
                  (test_case_data[i].get_failed_test_cases_count() < 1)):
            continue
        if mode.upper() == 'PAIRWISE':
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N,2)+1)))

            env = CIPairWiseEnv(test_case_data[i], conf)

        elif mode.upper() == 'POINTWISE':
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N,2)+1)))
            env = CIPointWiseEnv(test_case_data[i], conf)
        elif mode.upper() == 'LISTWISE':
            conf.max_test_cases_count = get_max_test_cases_count(test_case_data)
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N,2)+1)))
            env = CIListWiseEnv(test_case_data[i], conf)
        elif mode.upper() == 'LISTWISE2':
            conf.max_test_cases_count = get_max_test_cases_count(test_case_data)
            N = test_case_data[i].get_test_cases_count()
            steps = int(episodes * (N * (math.log(N,2)+1)))
        print("Training agent with replaying of cycle " + str(i) + " with steps " + str(steps))
        total_steps+=steps

        print("millions steps, millions steps",total_steps)
        if model_save_path:
            previous_model_path = model_save_path
        model_save_path = model_path + "/" + mode + "_" + algo + dataset_name + "_" + str(
            start_cycle) + "_" + str(i)



        import torch as th


        if first_round:
            environment = Environment.create(
                environment=env
            )
            if algo.upper() == 'A2C':
                agent = Agent.create(
                    agent='a2c', environment=environment, network=[
                        dict(type='flatten'),
                        dict(type='dense', size=64, activation='relu'),
                        dict(type='dense', size=64, activation='relu'),
                        dict(type='linear', size=2),
                    ],
                    discount=0.90, memory=10000, learning_rate=0.0001, batch_size=64
                )

            if algo.upper() == 'DDPG':
                agent = Agent.create(
                    agent='ddpg', environment=environment, network=[
                        dict(type='flatten'),
                        dict(type='dense', size=64, activation='relu'),
                        dict(type='dense', size=64, activation='relu'),
                        dict(type='linear', size=2),
                    ],
                    discount=0.90, memory=10000, learning_rate=0.0001, batch_size=64
                )
            if algo.upper() == 'DQN':
                agent = Agent.create(
                    agent='dqn', environment=environment, network=[
                        dict(type='flatten'),
                        dict(type='dense', size=64, activation='relu'),
                        dict(type='dense', size=64, activation='relu'),
                        dict(type='linear', size=2),
                    ],
                    discount=0.90, memory=10000, learning_rate=0.0001, batch_size=64
                )

            training_start_time = datetime.now()
            j=0
            k=0
            train_rew=0
            while j<steps:
                states = environment.reset()
                terminal = 0
                step = 0
                k=0
                while terminal == 0 and j<steps:

                    actions = agent.act(states=states)

                    states, terminal, reward = environment.execute(actions=actions)
                    train_rew += reward
                    agent.observe(terminal=terminal, reward=reward)
                    j=j+1
                    k=k+1
                   #agent.save(directory=model_save_path)
                    agent.save(directory='model-numpy', format='numpy', append='episodes')
            training_end_time = datetime.now()
            first_round = False
        else:
            environment = Environment.create(
                environment=env
            )
            #agent=Agent.load(directory=previous_model_path)
            agent=Agent.load(directory='model-numpy', format='numpy', environment=environment)
            j=0
            k=0
            train_rew=0
            training_start_time = datetime.now()
            while j<steps:
                states = environment.reset()
                terminal = 0
                # if step == 128:
                #    break
                step = 0
                k=0
                while terminal == 0 and j<steps:
                    actions = agent.act(states=states)
                    states, terminal, reward = environment.execute(actions=actions)
                    train_rew+=reward
                    agent.observe(terminal=terminal, reward=reward)
                    j=j+1
                    k=k+1
                    agent.save(directory='model-numpy', format='numpy', append='episodes')
            training_end_time = datetime.now()
        print("Training agent with replaying of cycle " + str(i) + " is finished")

        j = i+1 ## test trained agent on next cycles
        while (((test_case_data[j].get_test_cases_count() < 6)
               or ((conf.dataset_type == "simple") and (test_case_data[j].get_failed_test_cases_count() == 0) ))
               and (j < end_cycle)):
            #or test_case_data[j].get_failed_test_cases_count() == 0) \
            j = j+1
        if j >= end_cycle-1:
            break
        if mode.upper() == 'PAIRWISE':
            env_test = CIPairWiseEnv(test_case_data[j], conf)
        elif mode.upper() == 'POINTWISE':
            env_test = CIPointWiseEnv(test_case_data[j], conf)
        elif mode.upper() == 'LISTWISE':
            env_test = CIListWiseEnv(test_case_data[j], conf)

        test_time_start = datetime.now()
        test_case_vector,test_rewards = TPAgentUtil.test_agent(env=env_test, algo=algo, model_path=model_save_path,
                                                  mode=mode)
        test_time_end = datetime.now()
        test_case_id_vector = []

        for test_case in test_case_vector:
            test_case_id_vector.append(str(test_case['test_id']))
            cycle_id_text = test_case['cycle_id']
        if test_case_data[j].get_failed_test_cases_count() != 0:
            apfd = test_case_data[j].calc_APFD_ordered_vector(test_case_vector)
            apfd_optimal = test_case_data[j].calc_optimal_APFD()
            apfd_random = test_case_data[j].calc_random_APFD()
            apfds.append(apfd)
        else:
            apfd =0
            apfd_optimal =0
            apfd_random =0
        nrpa = test_case_data[j].calc_NRPA_vector(test_case_vector)
        nrpas.append(nrpa)
        test_time = millis_interval(test_time_start,test_time_end)
        training_time = millis_interval(training_start_time,training_end_time)
        print("Testing agent  on cycle " + str(j) +
              " resulted in APFD: " + str(apfd) +
              " , NRPA: " + str(nrpa) +
              " , optimal APFD: " + str(apfd_optimal) +
              " , random APFD: " + str(apfd_random) +
              " , # failed test cases: " + str(test_case_data[j].get_failed_test_cases_count()) +
              " , # test cases: " + str(test_case_data[j].get_test_cases_count()), flush=True)
        log_file.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "," + mode + "," + algo + ","
                       + Path(model_save_path).stem + "," +
                       str(episodes) + "," + str(steps) + "," + str(cycle_id_text) + "," + str(training_time) +
                       "," + str(test_time) + "," + str(conf.win_size) + "," +
                       str(test_case_data[j].get_test_cases_count()) + "," +
                       str(test_case_data[j].get_failed_test_cases_count()) + "," + str(apfd) + "," +
                       str(nrpa) + "," + str(apfd_random) + "," + str(apfd_optimal) +"," + str(test_rewards)+"," + str(train_rew)+"," + os.linesep)
        log_file_test_cases.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "," + mode + "," + algo + ","
                       + Path(model_save_path).stem + "," +
                       str(episodes) + "," + str(steps) + "," + str(cycle_id_text) + "," + str(training_time) +
                       "," + str(test_time) + "," + str(conf.win_size) + "," +
                                  ('|'.join(test_case_id_vector)) + os.linesep)
        if (len(apfds)):

            print(f"avrage apfd so far is {mean(apfds)} and the standard deviation is {numpy.std(apfds)}")
        if (len(nrpas)):
            print(f"avrage nrpas so far is {mean(nrpas)} the standard deviation is {numpy.std(nrpas)}")

        log_file.flush()
        log_file_test_cases.flush()
    log_file.close()
    log_file_test_cases.close()

def reportDatasetInfo(test_case_data:list):
    cycle_cnt = 0
    failed_test_case_cnt = 0
    test_case_cnt = 0
    failed_cycle = 0
    for cycle in test_case_data:
        if cycle.get_test_cases_count() > 5:
            cycle_cnt = cycle_cnt+1
            test_case_cnt = test_case_cnt + cycle.get_test_cases_count()
            failed_test_case_cnt = failed_test_case_cnt+cycle.get_failed_test_cases_count()
            if cycle.get_failed_test_cases_count() > 0:
                failed_cycle = failed_cycle + 1
    print(f"# of cycle: {cycle_cnt}, # of test case: {test_case_cnt}, # of failed test case: {failed_test_case_cnt}, "
          f" failure rate:{failed_test_case_cnt/test_case_cnt}, # failed test cycle: {failed_cycle}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN debugger')
    old_limit = sys.getrecursionlimit()
    print("Recursion limit:" + str(old_limit))
    sys.setrecursionlimit(1000000)
    # parser.add_argument('--traningData',help='tranind data folder',required=False)
    parser.add_argument('-m', '--mode', help='[pairwise,pointwise,listwise] ', required=False)#non
    parser.add_argument('-a', '--algo', help='[a2c,dqn,..]', required=False)#non
    parser.add_argument('-d', '--dataset_type', help='simple, enriched', required=False, default="simple")
    parser.add_argument('-e', '--episodes', help='Training episodes ', required=False)#non
    parser.add_argument('-w', '--win_size', help='Windows size of the history', required=False)
    parser.add_argument('-t', '--train_data', help='Train set folder', required=False)#non
    parser.add_argument('-f', '--first_cycle', help='first cycle used for training', required=False)
    parser.add_argument('-c', '--cycle_count', help='Number of cycle used for training', required=False)
    parser.add_argument('-l', '--list_size', help='Maximum number of test case per cycle', required=False)
    parser.add_argument('-o', '--output_path', help='Output path of the agent model', required=False)


    # parser.add_argument('-f','--flags',help='Input csv file containing testing result',required=False)
    supported_formalization = ['PAIRWISE', 'POINTWISE', 'LISTWISE']
    supported_algo = ['DQN', "A2C", "DDPG"]
    args = parser.parse_args()
    args.mode='pointwise'
    args.algo='ddpg'
    #args.output_path='C:/Users/phili/Documents/Paulina Old/tp_rl-master_kerass/tp_rl-master_keras/tp_rl/testCase_prioritization/model'iofrol-additional-featurespaintcontrol-additional-features
    args.dataset_type="enriched"
    args.episodes='200'
    args.train_data='C:/Users/phili/myrep_rl/drl_se/Test_case_prioritization_problem/data/Commons_math.csv'
    assert supported_formalization.count(args.mode.upper()) == 1, "The formalization mode is not set correctly"
    assert supported_algo.count(args.algo.upper()) == 1, "The formalization mode is not set correctly"

    conf = Config()
    conf.train_data = args.train_data
    conf.dataset_name = Path(args.train_data).stem
    if not args.win_size:
        conf.win_size = 10
    else:
        conf.win_size = int(args.win_size)
    if not args.first_cycle:
        conf.first_cycle = 0
    else:
        conf.first_cycle = int(args.first_cycle)
    if not args.cycle_count:
        conf.cycle_count = 9999999

    if not args.output_path:
        conf.output_path = '../experiments/' + args.mode + "/" + args.algo + "/" + conf.dataset_name + "_" \
                           + str(conf.win_size) + "/"
        conf.log_file = conf.output_path + args.mode + "_" + args.algo + "_" + \
                        conf.dataset_name + "_" + args.episodes + "_" + str(conf.win_size) + "_log.txt"
    else:
        conf.output_path = args. output_path + "/" + args.mode + "/" + args.algo + "/" + conf.dataset_name + "_" \
                           + str(conf.win_size) + "/"
        conf.log_file = conf.output_path + args.mode + "_" + args.algo + "_" + \
                        conf.dataset_name + "_" + args.episodes + "_" + str(conf.win_size) + "_log.txt"

test_data_loader = TestCaseExecutionDataLoader(conf.train_data, args.dataset_type)
test_data = test_data_loader.load_data()
ci_cycle_logs = test_data_loader.pre_process()
reportDatasetInfo(test_case_data=ci_cycle_logs)
conf.dataset_type = args.dataset_type

experiment(mode=args.mode, algo=args.algo.upper(), test_case_data=ci_cycle_logs, episodes=int(args.episodes),
           start_cycle=conf.first_cycle, verbos=False,
           end_cycle=conf.first_cycle + conf.cycle_count - 1, model_path=conf.output_path, dataset_name="", conf=conf)
