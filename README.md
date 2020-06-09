# Exploration via Hindsight Goal Generation

This is the TensorFlow implementation for the thesis Goal-Based Hindsight Goal Generation for Robotic Object Manipulation with Sparse-Reward Deep Reinforcement Learning (Matthias Brucker, 2020)
It is based on the implementation of the HGG paper [Exploration via Hindsight Goal Generation](http://arxiv.org/abs/1906.04279) accepted by NeurIPS 2019.



## Requirements
1. Ubuntu 16.04 (newer versions such as 18.04 should work as well)
2. Python 3.5.2 (newer versions such as 3.6.9 should work as well)
3. MuJoCo == 1.50.1.68
4. Have a look at requirements.txt for python package versions (e.g. pymujoco-py == 2.0.2.5, TensorFlow == 1.14). Install requirements with pip install -r requirements.txt
7. Install gym from https://github.com/mbrucker07/gym.git. Certain environment specifications and parameters are set there. 

## Running Commands from G-HGG

Run the following commands to reproduce results on FetchPushLabyrinth, FetchPickObstacle, FetchPickNoObstacle, and FetchPickAndThrow environments.

Training Results:

```bash
# FetchPushLabyrinth
# HER (with EBP)
python train.py --tag 000 --learn normal --env FetchPushLabyrinth-v1 --goal custom 
# HGG (with HER, EBP and STOP condition)
python train.py --tag 010 --learn hgg --env FetchPushLabyrinth-v1 --goal custom --stop_hgg_threshold 0.3
# G-HGG (with HER, EBP and STOP condition)
python train.py --tag 020 --learn hgg --env FetchPushLabyrinth-v1 --goal custom --graph True --n_x 31 --n_y 31 --n_z 11 --stop_hgg_threshold 0.3 

# FetchPickObstacle
python train.py --tag 100 --learn normal --env FetchPickObstacle-v1 --goal custom 
python train.py --tag 110 --learn hgg --env FetchPickObstacle-v1 --goal custom --stop_hgg_threshold 0.3
python train.py --tag 120 --learn hgg --env FetchPickObstacle-v1 --goal custom --graph True --n_x 31 --n_y 31 --n_z 11 --stop_hgg_threshold 0.3

# FetchPickNoObstacle
python train.py --tag 200 --learn normal --env FetchPickNoObstacle-v1 --goal custom 
python train.py --tag 210 --learn hgg --env FetchPickNoObstacle-v1 --goal custom --stop_hgg_threshold 0.3
python train.py --tag 220 --learn hgg --env FetchPickNoObstacle-v1 --goal custom --graph True --n_x 31 --n_y 31 --n_z 11 --stop_hgg_threshold 0.3

# FetchPickAndThrow
python train.py --tag 300 --learn normal --env FetchPickAndThrow-v1 --goal custom 
python train.py --tag 310 --learn hgg --env FetchPickAndThrow-v1 --goal custom --stop_hgg_threshold 0.9
python train.py --tag 320 --learn hgg --env FetchPickAndThrow-v1 --goal custom --graph True --n_x 51 --n_y 51 --n_z 7 --stop_hgg_threshold 0.9
```

## Plotting

To plot the agent's performance on multiple training runs, copy all training run directories into one directory. For example, we put all FetchPushLabyrinth runs in a directory called BA_Labyrinth, same for FetchPickObstacle (BA_Obstacle), FetchPickNoObstacle(BA_NoObstacle) and FetchPickAndThrow (BA_Throw). naming=1 is for our the result plots in the thesis, using naming=0 is generally applicable and recommended.

```bash
# Scheme: python plot.py log_dir env_id --naming <naming_code> --e_per_c <episodes per cycle>
python plot.py log/Test_Labyrinth FetchPushLabyrinth-v1 --naming 0 --e_per_c 20
```

## Figures

Figures and the data they are based on can be found in the directory "figures" and were generate with the following scripts:

```bash
# Result and Ablation plots (Figures are already generated in the respective subdirectories in directory "figures"):
./create_result_figures.sh

# Other plots (Figures are already generated in directory "figures"):
python create_figures.py
```

## Playing: 

To look at the agent solving the respective task according to his learned policy, issue the following command:

```bash
# Scheme: python play.py --env env_id --goal custom --play_path log_dir --play_epoch <epoch number, latest or best>
python play.py --env FetchPushLabyrinth-v1 --goal custom --play_path figures/BA_Labyrinth/000-ddpg-FetchPushLabyrinth-v1-hgg-mesh-stop/ --play_epoch best
python play.py --env FetchPickObstacle-v1 --goal custom --play_path figures/BA_Obstacle/100-ddpg-FetchPickObstacle-v1-hgg-mesh-stop/ --play_epoch best
python play.py --env FetchPickNoObstacle-v1 --goal custom --play_path figures/BA_NoObstacle/200-ddpg-FetchPickNoObstacle-v1-hgg-mesh-stop/ --play_epoch best
python play.py --env FetchPickAndThrow-v1 --goal custom --play_path figures/BA_Throw/300a-ddpg-FetchPickAndThrow-v1-hgg-mesh-stop/ --play_epoch best
```

## Running commands from HGG paper

Run the following commands to reproduce our main results shown in section 5.1 of the HGG paper.

```bash
python train.py --tag='HGG_fetch_push' --env=FetchPush-v1
python train.py --tag='HGG_fetch_pick' --env=FetchPickAndPlace-v1
python train.py --tag='HGG_hand_block' --env=HandManipulateBlock-v0
python train.py --tag='HGG_hand_egg' --env=HandManipulateEgg-v0
```
