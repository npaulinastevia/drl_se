# A Comparison of Reinforcement Learning Frameworks for Software Testing Tasks
This repository contains the training scripts, datasets, and experiments corresponding to our paper (link available soon)

## Requirements

Python 3.8+

## Set up
```
pip install -r requirements.txt
```
## Datasets.

The paper uses datasets from the papers [Reinforcement Learning for Test Case Prioritization](https://github.com/moji1/tp_rl) and 
[Wuji: Automatic Online Combat Game Testing Using Evolutionary Deep Reinforcement Learning](https://github.com/NeteaseFuxiRL/wuji).
Game_testing_problem\wuji\problem\mdp\netease\blockmaze contains the blockmaze environment and Test_case_prioritization_problem\data contains all the simple and enriched datasets discussed in the paper.

## Run the experiments.



Each directory contains the files we use to run the experiments.

Game_testing_problem directory contains game_Keras-rl.py, game_Stable=baselines.py and game_Tensorforce.py to run experiments involving the game testing problem

Test_case_prioritization_problem directory contains se_Keras.py, se_Tensorforce.py to run experiments involving the test case prioritization problem

For each file the options (run dqn or a2c) can be change directly on the file.

### Example.

```
python game_Keras-rl.py
```

to collect bugs on the blockmaze game by using the DQN algorithm from keras-rl framework.

## Hyperparameters lists

### Game Testing problem

stable_baselines3.dqn.DQN(
"MlpPolicy", env, learning_rate=0.00025, policy_kwargs = dict(net_arch=[256, 128, 128,4]), gamma=0.99, verbose=1, batch_size=128,
buffer_size=1000000, learning_starts=50000, tau=1.0,  train_freq=4, gradient_steps=1, 
replay_buffer_class=None, replay_buffer_kwargs=None, optimize_memory_usage=False, target_update_interval=10000, 
exploration_fraction=0.1, exploration_initial_eps=1.0, exploration_final_eps=0.05, max_grad_norm=10, 
tensorboard_log=None, policy_kwargs=None, seed=None, device='auto', _init_setup_model=True)

stable_baselines3.a2c.A2C(
"MlpPolicy", env, learning_rate=0.00025, n_steps=128, use_rms_prop=False, policy_kwargs=dict(net_arch=[256, 128, 128,4]), gamma=0.99, rms_prop_eps=1e-08
 n_steps=5,  gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5, 
max_grad_norm=0.5,  use_rms_prop=True, use_sde=False, sde_sample_freq=-1, normalize_advantage=False, 
tensorboard_log=None, policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True)

stable_baselines3.ppo.PPO(
"MlpPolicy", env, learning_rate=0.00025, n_steps=128, policy_kwargs= dict(net_arch=[256, 128, 128,4]), gamma=0.99, verbose=1,
batch_size=128, n_epochs=10,  gae_lambda=0.95, 
clip_range=0.2, clip_range_vf=None, normalize_advantage=True, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, 
sde_sample_freq=-1, target_kl=None, tensorboard_log=None, policy_kwargs=None, seed=None, device='auto', _init_setup_model=True)

tensorforce.agents.DeepQNetwork(
states, actions, memory,network=[
                      dict(type='flatten'),
                      dict(type='dense', size=256, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='linear', size=4),
                  ],
                  discount=0.99, memory=10000, learning_rate=0.00025, batch_size=128,
 max_episode_timesteps=None,  update_frequency=0.25, 
start_updating=None,  huber_loss=None, horizon=1,  reward_processing=None, return_processing=None, 
predict_terminal_values=False, target_update_weight=1.0, target_sync_frequency=1, state_preprocessing='linear_normalization', 
exploration=0.0, variable_noise=0.0, l2_regularization=0.0, entropy_regularization=0.0, config=None, saver=None, 
summarizer=None, tracking=None, recorder=None, **kwargs)

tensorforce.agents.AdvantageActorCritic(states, actions, 
parallel_interactions=4, network=[
                      dict(type='flatten'),
                      dict(type='dense', size=256, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='linear', size=4),
                  ],
                  discount=0.99, memory=10000, learning_rate=0.00025, batch_size=128
max_episode_timesteps=None,  use_beta_distribution=False, update_frequency=1.0,  horizon=1,  reward_processing=None, return_processing=None, 
advantage_processing=None, predict_terminal_values=False, critic='auto', critic_optimizer=1.0, state_preprocessing='linear_normalization', 
exploration=0.0, variable_noise=0.0, l2_regularization=0.0, entropy_regularization=0.0, config=None, saver=None,
summarizer=None, tracking=None, recorder=None, **kwargs)

tensorforce.agents.ProximalPolicyOptimization(states, actions, max_episode_timesteps,  
parallel_interactions=4, network=[
                      dict(type='flatten'),
                      dict(type='dense', size=256, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='linear', size=4),
                  ],
                  discount=0.99, memory=10000, learning_rate=0.00025, batch_size=128,
 use_beta_distribution=False,
 update_frequency=1.0,  multi_step=10, subsampling_fraction=0.33, likelihood_ratio_clipping=0.25, 
reward_processing=None, return_processing=None, advantage_processing=None, predict_terminal_values=False, baseline=None, 
baseline_optimizer=None, state_preprocessing='linear_normalization', exploration=0.0, variable_noise=0.0, l2_regularization=0.0, 
entropy_regularization=0.0,  config=None, saver=None, summarizer=None, tracking=None, recorder=None, **kwargs)

rl.agents.dqn.DQNAgent(model==[
                      dict(type='flatten'),
                      dict(type='dense', size=256, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='linear', size=4),
                  ], policy=LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                      value_max=1, value_min=0.05, value_test=.2, nb_steps=10000), 
memory=SequentialMemory(limit=10000, window_length=1),
                       gamma=.99, batch_size=128, enable_double_dqn=False
test_policy=None, enable_dueling_network=False, dueling_type='avg')

### Test Case prioritization problem

stable_baselines.a2c.A2C(policy, env, gamma=0.99, n_steps=5, vf_coef=0.25, ent_coef=0.01, max_grad_norm=0.5, learning_rate=0.0007, 
alpha=0.99, momentum=0.0, epsilon=1e-05, lr_schedule='constant', verbose=0, tensorboard_log=None, _init_setup_model=True, 
policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None)

stable_baselines.ddpg.DDPG(policy, env, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50, nb_rollout_steps=100,
nb_eval_steps=100, param_noise=None, action_noise=None, normalize_observations=False, tau=0.001, batch_size=128, 
param_noise_adaption_interval=50, normalize_returns=False, enable_popart=False, observation_range=(-5.0, 5.0), critic_l2_reg=0.0, 
return_range=(-inf, inf), actor_lr=0.0001, critic_lr=0.001, clip_norm=None, reward_scale=1.0, render=False, render_eval=False, 
memory_limit=None, buffer_size=50000, random_exploration=0.0, verbose=0, tensorboard_log=None, _init_setup_model=True,
policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1)

stable_baselines.deepq.DQN(policy, env, gamma=0.99, learning_rate=0.0005, buffer_size=50000, exploration_fraction=0.1, 
exploration_final_eps=0.02, exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True, learning_starts=1000, 
target_network_update_freq=500, prioritized_replay=False, prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, 
prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06, param_noise=False, n_cpu_tf_sess=None, verbose=0, 
tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None)

tensorforce.agents.AdvantageActorCritic(states, actions, 
parallel_interactions=1, network=[
                            dict(type='flatten'),
                            dict(type='dense', size=64, activation='relu'),
                            dict(type='dense', size=64, activation='relu'),
                            dict(type='linear', size=2),
                        ],
                        discount=0.99, memory=10000, learning_rate=0.0007, batch_size=32
max_episode_timesteps=None,  use_beta_distribution=False, update_frequency=1.0,  horizon=1,  reward_processing=None, return_processing=None, 
advantage_processing=None, predict_terminal_values=False, critic='auto', critic_optimizer=1.0, state_preprocessing='linear_normalization', 
exploration=0.0, variable_noise=0.0, l2_regularization=0.0, entropy_regularization=0.0, config=None, saver=None,
summarizer=None, tracking=None, recorder=None, **kwargs)

tensorforce.agents.DeepQNetwork(
states, actions, memory,network=[
                        dict(type='flatten'),
                        dict(type='dense', size=64, activation='relu'),
                        dict(type='dense', size=64, activation='relu'),
                        dict(type='linear', size=2),
                    ],
                    discount=0.99, memory=10000, learning_rate=0.0005, batch_size=32
 max_episode_timesteps=None,  update_frequency=0.25, 
start_updating=None,  huber_loss=None, horizon=1,  reward_processing=None, return_processing=None, 
predict_terminal_values=False, target_update_weight=1.0, target_sync_frequency=1, state_preprocessing='linear_normalization', 
exploration=0.0, variable_noise=0.0, l2_regularization=0.0, entropy_regularization=0.0, config=None, saver=None, 
summarizer=None, tracking=None, recorder=None, **kwargs)

tensorforce.agents.DeterministicPolicyGradient(states, actions, 
network=[
                        dict(type='flatten'),
                        dict(type='dense', size=64, activation='relu'),
                        dict(type='dense', size=64, activation='relu'),
                        dict(type='linear', size=1),
                    ],
                    discount=0.99, memory=10000, learning_rate=0.0005, batch_size=32
max_episode_timesteps=None,  
use_beta_distribution=True, update_frequency=1.0, start_updating=None,  horizon=1, 
reward_processing=None, return_processing=None, predict_terminal_values=False, critic='auto', critic_optimizer=1.0, 
state_preprocessing='linear_normalization', exploration=0.1, variable_noise=0.0, l2_regularization=0.0, entropy_regularization=0.0, 
parallel_interactions=1, config=None, saver=None, summarizer=None, tracking=None, recorder=None, **kwargs)

rl.agents.dqn.DQNAgent(model==[
                      dict(type='flatten'),
                      dict(type='dense', size=256, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='dense', size=128, activation='relu'),
                      dict(type='linear', size=4),
                  ], policy=LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_test=.05,value_min=0.02), 
memory=SequentialMemory(limit=10000, window_length=1),
                       gamma=.99, batch_size=32, enable_double_dqn=False
test_policy=None, enable_dueling_network=False, dueling_type='avg')

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Input(shape=env.observation_space.shape))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(number_actions))
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

rl.agents.ddpg.DDPGAgent( memory=SequentialMemory(limit=10000, window_length=1),gamma=0.99,  actor=actor, critic=critic,
                             critic_action_input=action_input,nb_steps_warmup_critic=100, nb_steps_warmup_actor=100)