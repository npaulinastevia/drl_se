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
import pickle
from rl.callbacks import ModelIntervalCheckpoint, FileLogger, TrainEpisodeLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate,InputLayer
from rl.agents.dqn import DQNAgent
from rl.agents.ddpg import DDPGAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from PairWiseEnv import CIPairWiseEnv,CIPairWiseEnvT
from ci_cycle import CICycleLog
from Config import Config
from TestcaseExecutionDataLoader import TestCaseExecutionDataLoader
from pathlib import Path
from CIListWiseEnv import CIListWiseEnv
from PointWiseEnv import CIPointWiseEnv
import sys


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

def test_agent(env: CIPairWiseEnv, model_path: str, algo, mode,agent):
        agent_actions = []

        print("Evaluation of an agent from " + model_path)
        agent.load_weights(f'{model_path}.h5f')
        model = agent
        if model:

            if mode.upper() == "PAIRWISE" and algo.upper() == "DQN":
                #env = model.get_env()
                obs = env.reset()
                done = False
                test_rew=0
                while True:
                #
                    #obs=obs.reshape((1,)+env.observation_space.shape)
                    #obs=obs[: ,None]
                    action = np.argmax(model.forward(obs))

                    obs, rewards, done, info = env.step(action)
                    test_rew+=rewards
                    if done:
                         break
                return env.sorted_test_cases_vector,test_rew
            elif mode.upper() == "POINTWISE":

                if model:
                    test_cases = env.cycle_logs.test_cases
                    #if algo.upper() != "dqn":
                       # env = DummyVecEnv([lambda: env])
                    #model.set_env(env)
                    obs = env.reset()
                    done = False
                    index = 0
                    test_cases_vector_prob = []
                    test_rew = 0
                    for index in range(0, len(test_cases)):
                        action= model.forward(obs)
                        obs, rewards, done, info = env.step(action)
                        test_rew += rewards
                        test_cases_vector_prob.append({'index': index, 'prob': action})
                        if done:
                            assert len(test_cases) == index + 1, "Evaluation is finished without iterating all " \
                                                                 "test cases "
                            break
                    test_cases_vector_prob = sorted(test_cases_vector_prob, key=lambda x: x['prob'],
                                                    reverse=False) ## the lower the rank, te higher the priority
                    sorted_test_cases = []
                    for test_case in test_cases_vector_prob:
                        sorted_test_cases.append(test_cases[test_case['index']])
                return sorted_test_cases,test_rew
                pass
            elif mode.upper() == "LISTWISE":
                if model:
                    test_cases = env.cycle_logs.test_cases
                    obs = env.reset()
                    done = False
                i=0
                test_rew =0
                while True and i<1000000:
                    i=i+1
                    action = np.argmax(model.forward(obs))
                    print(action)
                    if agent_actions.count(action) == 0 and action < len(test_cases):
                        if isinstance(action, list) or isinstance(action, np.ndarray):
                            agent_actions.append(action[0])
                        else:
                            agent_actions.append(action)
                        # print(len(agent_actions))

                    obs, rewards, done, info = env.step(action)

                    test_rew += rewards
                    if done:
                        break
                sorted_test_cases = []
                for index in agent_actions:
                    sorted_test_cases.append(test_cases[index])
                if ( i>= 1000000):
                    sorted_test_cases = test_cases
                return  sorted_test_cases,test_rew

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
    log_file.write("timestamp,mode,algo,model_name,episodes,steps,cycle_id,training_time,testing_time,winsize,test_cases,failed_test_cases, apfd, nrpa, random_apfd, optimal_apfd, accumulated_reward_test, accumulated_reward_train," + os.linesep)
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
        print("Training agent with replaying of cycle " + str(i) + " with steps " + str(steps))
        total_steps+=steps

        print("millions steps, millions steps",total_steps)
        if model_save_path:
            previous_model_path = model_save_path
        #model_save_path = model_path + "/" + mode + "_" + algo + dataset_name + "_" + str(
         #   start_cycle) + "_" + str(i)
        model_save_path = mode + "_" + algo + dataset_name + "_" + str(
            start_cycle) + "_" + str(i)


        if first_round:

            memory = SequentialMemory(limit=10000, window_length=1)
            training_start_time = datetime.now()
            if algo.upper()=='DQN':
                model = Sequential()
                model.add(Flatten(input_shape=((1,) + env.observation_space.shape)))
                model.add(Dense(64, activation="relu"))
                model.add(Dense(64, activation="relu"))
                nb_actions = env.action_space.n
                model.add(Dense(nb_actions))
                model.add(Activation('linear'))
                model.compile(
                    loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])
                policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_test=.05,value_min=0.02, nb_steps=steps)
                tp_agen = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,batch_size=32,
                                 enable_double_dqn=False, policy=policy)

            if algo.upper() == 'DDPG':
                nb_actions = env.action_space.shape[0]
                actor = Sequential()
                actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))

                actor.add(Input(shape=env.observation_space.shape))
                actor.add(Dense(64))
                actor.add(Activation('relu'))
                actor.add(Dense(64))
                actor.add(Activation('relu'))
                actor.add(Dense(nb_actions))
                actor.add(Activation('sigmoid'))

                action_input = Input(shape=(nb_actions,), name='action_input')
                observation_input =  Input(shape=(1,) + env.observation_space.shape, name='observation_input')
                flattened_observation = Flatten()(observation_input)
                x = Concatenate()([action_input, flattened_observation])
                x = Dense(64)(x)
                x = Activation('relu')(x)
                x = Dense(64)(x)
                x = Activation('relu')(x)
                x = Dense(1)(x)
                x = Activation('linear')(x)
                critic = Model(inputs=[action_input, observation_input], outputs=x)
                tp_agen = DDPGAgent(nb_actions=nb_actions,  memory=memory,gamma=0.99,  actor=actor, critic=critic,
                             critic_action_input=action_input,nb_steps_warmup_critic=100, nb_steps_warmup_actor=100)

            tp_agen.compile(Adam(lr=5e-4), metrics=['mae']
                            )
            tp_agen.fit(env, nb_steps=steps, visualize=False, verbose=2)
            tp_agen.save_weights(f'{model_save_path}.h5f', overwrite=True)
            training_end_time = datetime.now()
            first_round = False
        else:
            #tp_agent = TPAgentUtil.load_model(algo=algo, env=env, path=previous_model_path)
            tp_agen.load_weights(f'{previous_model_path}.h5f')
            training_start_time = datetime.now()
            tp_agen.fit(env, nb_steps=steps, visualize=False,verbose=2)
            tp_agen.save_weights(f'{model_save_path}.h5f', overwrite=True)
            training_end_time = datetime.now()

        print("Training agent with replaying of cycle " + str(i) + " is finished")

        j = i+1 ## test trained agent on next cycles
        while (((test_case_data[j].get_test_cases_count() < 6)
               or ((conf.dataset_type == "simple") and (test_case_data[j].get_failed_test_cases_count() == 0) ))
               and (j < end_cycle)):
            j = j+1
        if j >= end_cycle-1:
            break
        if mode.upper() == 'PAIRWISE':
            env_test = CIPairWiseEnvT(test_case_data[j], conf)
        elif mode.upper() == 'POINTWISE':
            env_test = CIPointWiseEnv(test_case_data[j], conf)
        elif mode.upper() == 'LISTWISE':
            env_test = CIListWiseEnv(test_case_data[j], conf)


        test_time_start = datetime.now()
        test_case_vector,test_rewards = test_agent(env=env_test, algo=algo, model_path=model_save_path,
                                                  mode=mode,agent=tp_agen)
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
                       str(nrpa) + "," + str(apfd_random) + "," + str(apfd_optimal) + "," +str(test_rewards)+ "," +"," +os.linesep)
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
    parser.add_argument('-m', '--mode', help='[pairwise,pointwise,listwise] ', required=False)
    parser.add_argument('-a', '--algo', help='[a2c,dqn,..]', required=False)
    parser.add_argument('-d', '--dataset_type', help='simple, enriched', required=False, default="simple")
    parser.add_argument('-e', '--episodes', help='Training episodes ', required=False)
    parser.add_argument('-w', '--win_size', help='Windows size of the history', required=False)
    parser.add_argument('-t', '--train_data', help='Train set folder', required=False)
    parser.add_argument('-f', '--first_cycle', help='first cycle used for training', required=False)
    parser.add_argument('-c', '--cycle_count', help='Number of cycle used for training', required=False)
    parser.add_argument('-l', '--list_size', help='Maximum number of test case per cycle', required=False)
    parser.add_argument('-o', '--output_path', help='Output path of the agent model', required=False)


    # parser.add_argument('-f','--flags',help='Input csv file containing testing result',required=False)
    supported_formalization = ['PAIRWISE', 'POINTWISE', 'LISTWISE']
    supported_algo = ['DQN', "DDPG"]
    args = parser.parse_args()
    args.mode='listwise'
    args.algo='dqn'
    args.dataset_type="simple"
    args.episodes='200'
    args.train_data='C:/Users/phili/myrep_rl/drl_se/Test_case_prioritization_problem/data/paintcontrol-additional-features.csv'
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
### open data



reportDatasetInfo(test_case_data=ci_cycle_logs)
from tensorflow.python.framework.ops import disable_eager_execution
tf.compat.v1.disable_eager_execution()

#training using n cycle staring from start cycle
conf.dataset_type = args.dataset_type

experiment(mode=args.mode, algo=args.algo.upper(), test_case_data=ci_cycle_logs, episodes=int(args.episodes),
           start_cycle=conf.first_cycle, verbos=False,
           end_cycle=conf.first_cycle + conf.cycle_count - 1, model_path=conf.output_path, dataset_name="", conf=conf)
# .. lets test this tommorow by passing args
