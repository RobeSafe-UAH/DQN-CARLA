# Deep-Q Learning
Deep Reinforcement Learning based on DQN using CARLA Simulator for self-driving vehicles.
![DQN architecture](media/DQN_architecture.jpg)
## Overview
This project uses CARLA Simulator and DQN algorithm to training agents which could predict
the following action to be taken by a self-driving vehicle in terms of control commands.
Several agents are contemplated in the project depending on the input data used to train the
Deep Reinforcement Learning algorithm. 
Both training and validation programs are available in the repository.

## Requirements
- Python3.6
- Numpy
- Tensorflow==1.14.0
- Keras==2.2.4
- OpenCV==4.1.2

This requirements will be installed automatically by using 
```ssh
pip3 install -e requirements.txt"
```

## Get Started and Usage
### config.py
Choose the desire settings in config.py before launch any training or play stage.
Make sure that the paths involve in the program exist, if not, create them to avoid failures in execution.
### src/gent_model.py
Settings for Neural Networks involve in training process.
### launch_DQN.py
In a terminal make:
```sh
python3 launch_DQN.py
```
## Results
Each training attempt generates a log file into logs folder where some metrics are saved in order to check how the process is working.
![DQN average reward](media/average_reward_DQN_CARLA_WP.tif)
![DQN average distance](media/average_dist_DQN_CARLA_WP.tif)
![DQN maximum reward](media/max_reward_DQN_CARLA_WP.tif)

## Video
![DQN gif](media/DQN_github.gif)

