# Bounding Box Based Hindsight Goal Generation

This is the TensorFlow implementation for the thesis Goal-Based Hindsight Goal Generation for Robotic Object Manipulation with Sparse-Reward Deep Reinforcement Learning (Matthias Brucker, 2020)
It is based on the implementation of the HGG paper [Exploration via Hindsight Goal Generation](http://arxiv.org/abs/1906.04279) accepted by NeurIPS 2019.



## Requirements
- Ubuntu (this code was tested on Ubuntu 18.04)
- MuJoCO (version 2.0)
- Pytorch (version 1.5.0)
- Tensorflow (version 1.14)
- other dependencies in requirements.txt

## Usage
First perform training of vision module.

Change to directory vae.

(1) Generate data 
```bash
python generate_data.py --env=FetchGenerativeEnv-v1  --task generate
```
(2) Train MONet network
```bash
python train_monet.py --env FetchGenerativeEnv-v1 --task train
```
(3) Prepare the data for training the BboxEncoder
```bash
python Bbox.py --env FetchGenerativeEnv-v1 --task prepare
```

(4) Train the BboxEncoder
```bash
python Bbox.py --env FetchGenerativeEnv-v1 --task train
```

(The next steps must be performed for each environment)

Run the index inference in each environment.
```bash
python index_inference.py --env=<name-environment> --vae_dist_help True --vae_type bbox --img_size 64
```

Train the agent:
```bash
#with example of parameter used in FetchPushMovingComEnv-v2
# HGG (with HER, EBP and STOP condition)
python train.py --tag 010 --learn hgg --env=FetchPushMovingComEnv-v2 --goal interval --graph False --stop_hgg_threshold 0.3 --vae_dist_help False --img_size 64 --show_goals 40 --epoches 25
# Bbox-HGG (with HER, EBP and STOP condition)
python train.py --tag 020 --learn hgg --env=FetchPushMovingComEnv-v2 --goal intervalPAVRewMod --rew_mod_val -2 --graph False --stop_hgg_threshold 0.3 --vae_dist_help True --vae_type bbox --img_size 64 --dist_estimator_type subst --show_goals 40 --epoches 25
# Bbox-HGG (with HER, EBP and STOP condition, with CTR)
python train.py --tag 030 --learn hgg --env=FetchPushMovingComEnv-v2 --goal intervalPAVRewMod --rew_mod_val -2 --graph False --stop_hgg_threshold 0.3 --vae_dist_help True --vae_type bbox --img_size 64 --dist_estimator_type subst --show_goals 40 --epoches 25 --imaginary_obstacle_transitions True --im_buffer_size 100 --im_warmup 80 --im_n_per_type 3
# Bbox-HGG (with HER, EBP and STOP condition, with real coordinates from simulation (no index inference needed))
python train.py --tag 040 --learn hgg --env=FetchPushMovingComEnv-v2 --goal intervalPAVRewMod --rew_mod_val -3 --graph False --stop_hgg_threshold 0.3 --vae_dist_help False --img_size 64 --dist_estimator_type substReal --show_goals 40 --epoches 25
# Bbox-HGG (with HER, EBP and STOP condition, with CTR, with real coordinates from simulation (no index inference needed))
python train.py --tag 050 --learn hgg --env=FetchPushMovingComEnv-v2 --goal intervalPAVRewMod --rew_mod_val -3 --graph False --stop_hgg_threshold 0.3 --vae_dist_help False --img_size 64 --dist_estimator_type substReal --show_goals 40 --epoches 25 --imaginary_obstacle_transitions True --im_buffer_size 100 --im_warmup 80 --im_n_per_type 3
```

Obstacle avoidance test:
```bash
#with example of parameter used in FetchPushMovingComEnv-v2
# HGG
python after_train_tester.py --env=FetchPushMovingComEnv-v2  --goal intervalTest --play_path <path-to-folder-with-agent-weights> --play_epoch best --img_size 64
# Bbox-HGG
python after_train_tester.py --env=FetchPushMovingComEnv-v2  --goal intervalTestExtendedPAV --play_path <path-to-folder-with-agent-weights> --play_epoch best --vae_dist_help True --vae_type bbox --img_size 64
# Bbox-HGG (with real coordinates from simulation)
python after_train_tester.py --env=FetchPushMovingComEnv-v2  --goal intervalTestExtendedPAV --play_path <path-to-folder-with-agent-weights> --play_epoch best --img_size 64
```
'<path-to-folder-with-agent-weights>' is for example: log\030-ddpg-FetchPushMovingComEnv-v2-intervalPAVRewMod-hgg-stop-subst-bbox-rewmodVal(-2.0)-IMAGINARY
