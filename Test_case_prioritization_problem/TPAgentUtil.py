
import keras
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, InputLayer, Reshape
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate,InputLayer
import numpy as np

from PairWiseEnv import CIPairWiseEnv, CIPairWiseEnvT
from PointWiseEnv import CIPointWiseEnv

class TPAgentUtil:

    supported_algo = ['DQN',"ACKTR", "DDPG"]

    def create_model(algo, env):
        assert TPAgentUtil.supported_algo.count(algo.upper()) == 1, "The algorithms  is not supported for" \
                                                                    " pairwise formalization"
        if algo.upper() == "DQN":
            #--- here starts theold dqn implementation
            #from stable_baselines3.dqn import MlpPolicy
            #from stable_baselines3 import DQN
            #model = DQN(MlpPolicy, env, gamma=0.90, learning_rate=0.0005, buffer_size=10000,
            #            exploration_fraction=1, exploration_final_eps=0.02, exploration_initial_eps=1.0,
            #           train_freq=1, batch_size=32, learning_starts=1000,
            #           target_update_interval=500, verbose=0,
            #           tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
            #         seed=None)

            #following is the create model with keras rl
            nb_actions = env.action_space.n
            model = Sequential()
            model.add(Flatten(input_shape=((1,)+env.observation_space.shape)))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(nb_actions))
            model.add(Activation('linear'))
            model.compile(
                 loss="categorical_crossentropy",
                 optimizer="adam",
                 metrics=["accuracy"])

        return model

    def load_model(algo, env, path):
        if algo.upper() == "DQN":
            #here is what we have from the previous DQN set upwith SB3
            #from stable_baselines3.dqn import MlpPolicy
            #from stable_baselines3 import DQN
            #model = DQN.load(path)
            #model.set_env(env)
            #following keras rl is used
            nb_actions = env.action_space.n
            from rl.agents.ddpg import DDPGAgent
            model = Sequential()
            model.add(Flatten(input_shape=((1,)+env.observation_space.shape)))
            #
            #
            #
            #model.add(InputLayer(batch_input_shape=env.observation_space.shape))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(64, activation="relu"))
            # model.add(Dense(64, activation="relu"))
            #
            model.add(Dense(nb_actions))
            model.add(Activation('linear'))
            model.compile(
                 loss="categorical_crossentropy",
                 optimizer="adam",
                 metrics=["accuracy"])
            #
            memory = SequentialMemory(limit=50000, window_length=1)
            policy = BoltzmannQPolicy()
            tp_agen = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                               target_model_update=1e-2, enable_double_dqn=True, enable_dueling_network=False,
                               policy=policy)
            #
            #
            #
            #
            tp_agen.compile(Adam(lr=1e-4), metrics=['mae']
                            )
            tp_agen.load_weights(f'{path}.h5f')
            model=tp_agen
            #following tensorforce is used
            #from tensorforce.agents import Agent

            #model = Agent.load(directory=f'{path}')
        elif algo.upper() == "PPO2":
            from stable_baselines.ppo2 import PPO2
            model = PPO2.load(path)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        elif algo.upper() == "A2C":
            from stable_baselines.a2c import A2C
            model = A2C.load(path)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        elif algo.upper() == "ACER":
            from stable_baselines.acer import ACER
            model = ACER.load(path)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        elif algo.upper() == "ACKTR":
            from stable_baselines.acktr import ACKTR
            model = ACKTR.load(path)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        elif algo.upper() == "GAIL":
            import stable_baselines.gail
        elif algo.upper() == "HER":
            import stable_baselines.her
            pass
        elif algo.upper() == "PPO1":
            from stable_baselines.ppo1 import PPO1
            model = PPO1.load(path)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        elif algo.upper() == "TRPO":
            from stable_baselines.trpo_mpi import TRPO
            model = TRPO.load(path)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        elif algo.upper() == "DDPG":
            pass
            # here is what we have from the previous DQN set upwith SB3
            # from stable_baselines3.dqn import MlpPolicy
            # from stable_baselines3 import DQN
            # model = DQN.load(path)
            # model.set_env(env)
            # following keras rl is used
            nb_actions = env.action_space.shape[0]
            from rl.agents.ddpg import DDPGAgent
            # model = Sequential()
            # model.add(Flatten(input_shape=((1,)+env.observation_space.shape)))
            #
            #
            #
            # #model.add(InputLayer(batch_input_shape=env.observation_space.shape))
            # model.add(Dense(64, activation="relu"))
            # model.add(Dense(64, activation="relu"))
            # model.add(Dense(64, activation="relu"))
            #
            # model.add(Dense(nb_actions))
            # model.add(Activation('linear'))
            # model.compile(
            #     loss="categorical_crossentropy",
            #     optimizer="adam",
            #     metrics=["accuracy"])
            #
            # memory = SequentialMemory(limit=50000, window_length=1)
            # policy = BoltzmannQPolicy()
            #
            #
            #
            #
            actor = Sequential()
            actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))

            #actor.add(Input(shape=env.observation_space.shape))
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
            memory = SequentialMemory(limit=10000, window_length=1)
            critic = Model(inputs=[action_input, observation_input], outputs=x)
            tp_agen = DDPGAgent(nb_actions=nb_actions,  memory=memory,gamma=0.99,  actor=actor, critic=critic,
                                critic_action_input=action_input,nb_steps_warmup_critic=100, nb_steps_warmup_actor=10)
            tp_agen.compile(Adam(lr=1e-4), metrics=['mae']
                            )
            tp_agen.load_weights(f'{path}.h5f')
            model=tp_agen

            # following tensorforce is used
            # from tensorforce.agents import Agent

            # model = Agent.load(directory=f'{path}')
        elif algo.upper() == "TD3":
            from stable_baselines import TD3
            from stable_baselines.td3.policies import MlpPolicy
            model = TD3(MlpPolicy, env, verbose=0)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        elif algo.upper() == "SAC":
            from stable_baselines.sac.policies import MlpPolicy
            from stable_baselines import SAC
            model = SAC(MlpPolicy, env, verbose=0)
            env = DummyVecEnv([lambda: env])
            model.set_env(env)
        else:
            return None
        return model

    def test_agent(env: CIPairWiseEnv, model_path: str, algo, mode):
        agent_actions = []

        print("Evaluation of an agent from " + model_path)

        #model = TPAgentUtil.load_model(path=model_path, algo=algo, env=env)
        from tensorforce.environments import Environment
        from tensorforce.agents import Agent
        environment = Environment.create(
            environment=env
        )
        # agent = Agent.create(
        #     agent='ac', environment=environment, parallel_interactions=4, network=[
        #         dict(type='flatten'),
        #         dict(type='dense', size=64, activation='relu'),
        #         dict(type='dense', size=64, activation='relu'),
        #         dict(type='linear', size=2),
        #     ],
        #     discount=0.99, memory=10000, learning_rate=0.0001, batch_size=128,
        #
        # )
        # nb_actions = env.action_space.n
        # model = Sequential()
        # model.add(Flatten(input_shape=((1,) + env.observation_space.shape)))
        # #
        # #
        # #
        # # model.add(InputLayer(batch_input_shape=env.observation_space.shape))
        # model.add(Dense(64, activation="relu"))
        # model.add(Dense(64, activation="relu"))
        # # model.add(Dense(64, activation="relu"))
        # #
        # model.add(Dense(nb_actions))
        # model.add(Activation('linear'))
        # model.compile(
        #     loss="categorical_crossentropy",
        #     optimizer="adam",
        #     metrics=["accuracy"])
        # #
        # memory = SequentialMemory(limit=50000, window_length=1)
        # policy = BoltzmannQPolicy()
        # tp_agen = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
        #                    target_model_update=1e-2, enable_double_dqn=True, enable_dueling_network=False,
        #                    policy=policy)
        # #
        # #
        # #
        # #
        # tp_agen.compile(Adam(lr=1e-4), metrics=['mae']
        #                 )
        # tp_agen.load_weights(f'{model_path}.h5f')
        # model = tp_agen
        #model = Agent.load(directory=model_path)
        model = Agent.load(directory='model-numpy', format='numpy', environment=environment)
        if model:

            if mode.upper() == "PAIRWISE" and algo.upper() != "DQN" :
                #env = model.get_env()
                obs = env.reset()
                states=env.reset()
                #env.test_cases_vector()

                done = 0
                step_t=0
                test_rew=0
                while done==0 and step_t<300:
                    step_t+=1
                    actions= model.act(states=states)

                    obs, rewards, done, info = env.step(actions)
                    test_rew+=rewards
                    #states, terminal, reward = environment.execute(actions=actions)

                    model.observe(terminal=done, reward=rewards)

                    if done:
                        break
                #return env.get_attr("sorted_test_cases_vector")[0]


                return env.sorted_test_cases_vector,test_rew
            elif mode.upper() == "PAIRWISE" and algo.upper() == "DQN":
                #env = model.get_env()
                #KERAS RL
                obs = env.reset()
                #
                done = False
                test_rew=0
                while True:
                #
                    #obs=obs.reshape((1,)+env.observation_space.shape)
                    #obs=obs[: ,None]
                #
                #

                    action = model.act(states=obs)

                    #action = np.argmax(model.forward(obs))

                    obs, rewards, done, info = env.step(action)
                    model.observe(terminal=done, reward=rewards)
                    test_rew+=rewards
                    if done:
                         break


                    # Initialize episode
                # states = env.reset()
                # terminal = False

                # while not terminal:
                        # Episode timestep
                #      actions = model.act(states=states)

                #     states,  terminal, reward = env.execute(actions)
                #        model.observe(terminal=terminal, reward=reward)
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
                        #action= model.forward(obs)
                        action=model.act(states=obs)
                        obs, rewards, done, info = env.step(action)
                        test_rew += rewards
                        model.observe(terminal=done, reward=rewards)
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
                    #if algo.upper() != "DQN":
                     #   env = DummyVecEnv([lambda: env])
                    # model.set_env(env)
                    obs = env.reset()
                    done = False
                i=0
                test_rew =0
                while True and i<10000:
                    i=i+1
                    # action, _states = model.predict(obs, deterministic=False)
                    #action = model.forward(obs)
                    action = model.act(states=obs)


                    #print(action)
                    #print(len(agent_actions))
                    if agent_actions.count(action) == 0 and action < len(test_cases):
                        if isinstance(action, list) or isinstance(action, np.ndarray):
                            agent_actions.append(action[0])
                        else:
                            agent_actions.append(action)
                        # print(len(agent_actions))

                    obs, rewards, done, info = env.step(action)
                    model.observe(terminal=done, reward=rewards)
                    test_rew += rewards
                    if done:
                        break
                sorted_test_cases = []

                for index in agent_actions:
                    sorted_test_cases.append(test_cases[index])
                if ( i>= 10000):
                    sorted_test_cases = test_cases
                return  sorted_test_cases,test_rew
            elif mode.upper() == "LISTWISE2":
                if model:
                    env = model.get_env()
                    obs = env.reset()
                    action, _states = model.predict(obs, deterministic=True)
                    env.step(action)
                    if algo.upper() != "DQN":
                        return env.get_attr("sorted_test_cases")[0]
                    else:
                        return env.sorted_test_cases


