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