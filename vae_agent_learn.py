from custom_start import get_args_and_initialize
import time
from envs import make_env
from agents.ou_noise import OUNoise
from agents.actor_critic import TD3
from temp_func import conc_latent
from gym.envs.robotics.fetch.push_labyrinth3 import goal_distance#todo import from more general part
from copy import copy, deepcopy
from algorithm.replay_buffer import Trajectory, goal_concat, ReplayBuffer_Episodic
from vae.vae_torch import setup_distance_ae_and_trainer
from utils.transforms_utils import extend_obs, BA
from agents.distance_estimator import DistanceEstimator
from distance_evaluation import process_distances, create_dist_images, \
    create_rollout_video_with_distances, write_distances_to_csv, create_rollout_video
from collections import namedtuple


AGENT_CHECKPOINT_NAME = 'td3_simplified'#'td3_no_obstacle'
AGENT_LAST = 'td3_simplified_last'#'td3_no_obstacle_last'
ESTIMATOR1_CHECKPOINT_NAME = 'dist_estimator_simplified'#'dist_estimator_no_obstacle'
ESTIMATOR2_CHECKPOINT_NAME = 'dist_estimator2_simplified'#'dist_estimator2_no_obstacle'
ESTIMATOR1_LAST = 'dist_estimator_simplified_last'#'dist_estimator_no_obstacle_last'
ESTIMATOR2_LAST = 'dist_estimator2_simplified_last'#'dist_estimator2_no_obstacle_last'
CSV_VID_PATTERN = 'dist_simplified_epoch_{}_it_{}'#'dist_no_obstacle_epoch_{}_it_{}'

#todo the env is generating some point outside the table/ before initiating env verify this is not the case
def hierarchical_training_loop(start_epoch, env, agent, replay_buffer, ae, trainer_ae, dist_estimator, dist_estimator2,
                               args, ou_noiser=None):
    if start_epoch > 0:#retraining; replay buffer must be filled again
        while replay_buffer.steps_counter < args.min_replay_size:
            agent_episode(0, 0, env, agent, replay_buffer, ae, args, ou_noiser, do_train=False)
    for epoch in range(start_epoch, args.training_epochs):
        #TODO HGG sample goals for each episode and update hgg sampler to choose goals to explore
        start_time = time.time()
        successes = 0
        for episode in range(args.training_episodes):
            #let the agent train in the environment
            mean_cr1_loss, mean_cr2_loss, mean_act_loss, done = agent_episode(epoch, episode, env, agent,
                                                                              replay_buffer, ae, args, ou_noiser)
            successes += float(done)
            #train the distance measure
            if epoch <= 200:
                freq = 50
            else:
                freq = 30
            if ((epoch >= 1 and epoch % freq == 0) or epoch==args.training_epochs-1) \
                    and (episode % 15 or episode==args.training_episodes-1):
                #create Replay Buffer
                for _ in range(3):
                    buffer_args = BA('none', 6)
                    buffer = ReplayBuffer_Episodic(buffer_args)
                    for i in range(6):
                        agent_episode(epoch, i, env, agent, buffer, ae, args, ou_noiser, deterministic_sampling=True)
                    batch_trajectories = buffer.buffer['obs']
                    ld1 = dist_estimator.train_with_labels(batch_trajectories, ae, trainer_ae)

                    b = replay_buffer.sample_for_distance(10, goal_key=dist_estimator2.goal_key,
                                                          goal_her_key=dist_estimator2.goal_her_key,
                                                          state_key=dist_estimator2.state_key)
                    ld2 = dist_estimator2.train_rl(b)
                args.csv_logger.add_log('estimator_loss', ld1)
                args.csv_logger.add_log('estimator_rl_loss', ld2)
                dist_estimator.save_train_checkpoint(ESTIMATOR1_CHECKPOINT_NAME, epoch)
                dist_estimator2.save_train_checkpoint(ESTIMATOR2_CHECKPOINT_NAME, epoch)
            args.csv_logger.finish_episode_log(episode)#this was in agent loop

        time_needed = time.time() - start_time
        print('finished epoch ', epoch)
        print('last actor loss ', mean_act_loss)
        print('last_critic1 loss ', mean_cr1_loss)
        print('last_critic2 loss ', mean_cr2_loss)
        print('success rate', successes/args.training_episodes)
        print('time needed ', time_needed)
        print('______________________________')
        args.csv_logger.add_log('time', time_needed)
        args.csv_logger.finish_epoch_log(epoch)
        #if epoch % args.test_interval == 0:
        #    test_agent(epoch,env, agent, ae, args)
        if epoch % 100 == 0 or epoch in [ 30, 80, 120, 180,230, 250, 380] or epoch == args.training_epochs-1:
            for i in range(2):#just to have more samples
                buffer_args = BA('none', 1)
                buffer = ReplayBuffer_Episodic(buffer_args)
                agent_episode(epoch, 0, env, agent, buffer, ae, args, ou_noiser, deterministic_sampling=True)
                trajectory = buffer.buffer['obs'][0]
                distance_dict, keys_order = process_distances(trajectory=trajectory,
                                                              dist_estimator=dist_estimator,
                                                              dist_estimator2=dist_estimator2, args=args)
                name = CSV_VID_PATTERN.format(epoch, i)
                write_distances_to_csv(distance_dict,name,args)
                dist_ims = create_dist_images(distance_dict, keys_order, args)
                create_rollout_video_with_distances(args, trajectory, dist_ims, name)
    agent.save(AGENT_LAST)
    dist_estimator.save(ESTIMATOR1_LAST)
    dist_estimator2.save(ESTIMATOR2_LAST)

def agent_episode(epoch, episode, env, agent, replay_buffer, ae, args, ou_noiser,
                  deterministic_sampling=False,do_train=True):
    #TODO HGG sample goals for each episode and update hgg sampler to choose goals to explore
    ae.eval()
    obs = env.reset()
    obs = extend_obs(ae, obs, args)
    first_obs = copy(obs)
    ou_noiser.reset()
    ##sample hgg goal for episode
    if args.goal_type == 'latent':
        goal = obs['goal_latent']
    elif args.goal_type == 'goal_space':
        goal = obs['desired_goal']
    elif args.goal_type == 'state':
        goal = obs['goal_state']
    elif args.goal_type == 'concat':
        goal = conc_latent(obs['goal_state'], obs['goal_latent'])
    else:
        raise Exception('goal obs type not valid')
    episode_reward = 0
    cr1_loss, cr2_loss, act_loss = None, None, None
    current = Trajectory(obs)
    for step in range(args.env_steps):
        if args.observation_type == 'latent':
            state = obs['state_latent']
        elif args.observation_type == 'real':
            state = obs['observation']
        elif args.observation_type == 'concat':
            state = conc_latent(obs['observation'], obs['state_latent'])
        else:
            raise Exception('observation type not valid')
        action = agent.get_action(state, goal)
        if not deterministic_sampling:
            action = ou_noiser.get_action(action, step)
        next_obs, reward, done, _ = env.step(action)
        next_obs = extend_obs(ae, next_obs, args)
        episode_reward += reward
        current.store_step(action, next_obs, reward, done)
        if deterministic_sampling:
            if done or step == args.env_steps - 1 or not next_obs['gripper_visible'] \
                    or next_obs['object_below_table']:#todo make last obs as done???
                replay_buffer.store_trajectory(current)
                break
        else:
            if done or step == args.env_steps - 1:#todo make last obs as done???
                replay_buffer.store_trajectory(current)
                if done:
                    if args.counter_achieved_goals % 50 == 0:
                        print('achieved goal!')
                        buffer_args = BA('none', 1)
                        buffer = ReplayBuffer_Episodic(buffer_args)
                        buffer.store_trajectory(deepcopy(current))
                        trajectory = buffer.buffer['obs'][0]
                        create_rollout_video(args, trajectory, 'achieved_goal_epoch_{}_episode_{}'.format(epoch, episode))
                    args.counter_achieved_goals +=1
                break
        obs = next_obs
    # TRAINING OF AGENT
    if not deterministic_sampling and do_train:
        if replay_buffer.steps_counter >= args.min_replay_size:
            for training_step in range(args.training_steps):
                batch = replay_buffer.sample_batch(args.batch_size, using_goal_as_field=True)
                c1l, c2l, al = agent.train_with_batch(batch)
                cr1_loss = cr1_loss + c1l if cr1_loss is not None else c1l
                cr2_loss = cr2_loss + c2l if cr2_loss is not None else c2l
                if al is not None:
                    act_loss = act_loss + al if act_loss is not None else al
        mean_cr1_loss = cr1_loss / float(args.training_steps) if cr1_loss is not None else 0
        mean_cr2_loss = cr2_loss / float(args.training_steps) if cr2_loss is not None else 0
        mean_act_loss = act_loss / float(args.training_steps) if act_loss is not None else 0
        args.csv_logger.add_log('critic1_loss', mean_cr1_loss)
        args.csv_logger.add_log('critic2_loss', mean_cr2_loss)
        args.csv_logger.add_log('actor_loss', mean_act_loss)
        args.csv_logger.add_log('episode_reward', episode_reward)
        args.csv_logger.add_log('episode_steps', step + 1)
        args.csv_logger.add_log('success', float(done))
        distance = goal_distance(next_obs['achieved_goal'], first_obs['desired_goal'])
        args.csv_logger.add_log('distance_to_goal', distance)
        #
        #args.csv_logger.finish_episode_log(episode)
        if epoch % args.checkpoint_interval == 0:
            #agent.save_train_checkpoint('td3_simplified', epoch)
            agent.save_train_checkpoint(AGENT_CHECKPOINT_NAME, epoch)
        return mean_cr1_loss, mean_cr2_loss, mean_act_loss, done

def test_agent(epoch, env, agent,ae, args):
    b = ReplayBuffer_Episodic(args)
    for e in range(10):
        agent_test_episode(env, agent, b,ae, args)
        args.test_csv_logger.finish_episode_log(e)
    batch = b.sample_batch(args.batch_size, using_goal_as_field=True)
    criticQ1_loss, criticQ2_loss, actor_loss = agent.evaluate_losses(batch)
    args.test_csv_logger.add_log('critic1_loss', criticQ1_loss)
    args.test_csv_logger.add_log('critic2_loss', criticQ2_loss)
    args.test_csv_logger.add_log('actor_loss', actor_loss)
    args.test_csv_logger.finish_epoch_log(epoch)
    del b

def agent_test_episode(env, agent, replay_buffer, ae, args):
    ae.eval()
    obs = env.reset()
    obs = extend_obs(ae, obs, args)
    first_obs = copy(obs)
    ##sample hgg goal for episode
    if args.goal_type == 'latent':
        goal = obs['goal_latent']
    elif args.goal_type == 'goal_space':
        goal = obs['desired_goal']
    elif args.goal_type == 'state':
        goal = obs['goal_state']
    elif args.goal_type == 'concat':
        goal = conc_latent(obs['goal_state'], obs['goal_latent'])
    else:
        raise Exception('goal obs type not valid')
    episode_reward = 0
    current = Trajectory(obs)
    for step in range(args.env_steps):
        if args.observation_type == 'latent':
            state = obs['state_latent']
        elif args.observation_type == 'real':
            state = obs['observation']
        elif args.observation_type == 'concat':
            state = conc_latent(obs['observation'], obs['state_latent'])
        else:
            raise Exception('observation type not valid')
        action = agent.get_action(state, goal)
        next_obs, reward, done, _ = env.step(action)
        next_obs = extend_obs(ae, next_obs, args)
        episode_reward += reward
        current.store_step(action, next_obs, reward, done)
        if done or step == args.env_steps - 1:#todo make last obs as done???
            replay_buffer.store_trajectory(current)
        obs = next_obs

    args.test_csv_logger.add_log('episode_reward', episode_reward)
    args.test_csv_logger.add_log('episode_steps', step + 1)
    args.test_csv_logger.add_log('success', float(done))
    distance = goal_distance(next_obs['achieved_goal'], first_obs['desired_goal'])
    args.test_csv_logger.add_log('distance_to_goal', distance)

def setup(args, recover_filename=None):
    env = make_env(args)
    args.compute_reward = env.compute_reward
    args.goal_distance = env.env.env.goal_distance
    args.counter_achieved_goals = 0
    replay_buffer = ReplayBuffer_Episodic(args)
    sample_obs = env.reset()
    latent_dim = args.latent_dim
    obs_dim = sample_obs['observation'].shape[0]
    goal_space_dim = sample_obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]
    if args.goal_type == 'latent':
        goal_dim = latent_dim
    elif args.goal_type == 'goal_space':
        goal_dim = goal_space_dim
    elif args.goal_type == 'state':
        goal_dim = obs_dim
    elif args.goal_type == 'concat':
        goal_dim = obs_dim+latent_dim
    else:
        raise Exception('goal obs type not valid')
    if args.observation_type == 'latent':
        state_dim = latent_dim
    elif args.observation_type == 'real':
        state_dim = obs_dim
    elif args.observation_type == 'concat':
        state_dim = obs_dim+latent_dim
    else:
        raise Exception('observation type not valid')

    agent = TD3(state_dim=state_dim, action_dim=action_dim, goal_dim=goal_dim,
                max_action=env.action_space.high, args=args)
    if recover_filename is not None:
        agent.load(filename=recover_filename)
    ou_noiser = OUNoise(env.action_space)
    #VAE
    ae, trainer_ae = setup_distance_ae_and_trainer(args, recover_filename=args.ae_recover_filename,compat_extra=True)
    return env, agent, ou_noiser, replay_buffer, ae, trainer_ae, state_dim, goal_dim, action_dim


if __name__ == "__main__":
    args = get_args_and_initialize()
    env, agent, ou_noiser, replay_buffer, ae, trainer_ae, \
    state_dim, goal_dim, action_dim = setup(args, recover_filename='td3_simplified_320')
    dist_estimator = DistanceEstimator(output_dim=1, state_dim=state_dim, goal_dim=goal_dim, min_range=[0],
                                       max_range=[200], observation_type=args.observation_type,
                                       goal_type=args.goal_type, args=args)
    dist_estimator.load('dist_estimator_simplified_300')
    dist_estimator2 = DistanceEstimator(output_dim=1, state_dim=state_dim, goal_dim=goal_dim, min_range=[0],
                                       max_range=[200], observation_type=args.observation_type,
                                       goal_type=args.goal_type, args=args)
    dist_estimator2.load('dist_estimator2_simplified_300')
    hierarchical_training_loop(start_epoch=321, env=env, agent=agent, replay_buffer=replay_buffer, ae=ae,
                               trainer_ae=trainer_ae, dist_estimator=dist_estimator, dist_estimator2=dist_estimator2,
                               ou_noiser=ou_noiser, args=args)