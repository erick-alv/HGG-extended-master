from custom_start import get_args_and_initialize
import time
from envs import make_env
from agents.ou_noise import OUNoise
from agents.actor_critic import TD3
from copy import copy, deepcopy
from algorithm.replay_buffer import Trajectory, ReplayBuffer_Episodic
from utils.transforms_utils import extend_obs, BA
from distance_evaluation import create_rollout_video, TrajectoryBufferForVid
import numpy as np
import psutil
from last_mem_check import make_the_last_call


AGENT_CHECKPOINT_NAME = 'td3_dyn'
AGENT_LAST = 'td3_dyn_last'
AGENT_BEST = 'td3_dyn_best'
RPL_FN = 'rpl_dyn'
CSV_VID_PATTERN = 'td3_dyn_epoch_{}_it_{}'#'dist_no_obstacle_epoch_{}_it_{}'

def main_loop(start_epoch, env, agent, replay_buffer, args, ou_noiser=None):
    if start_epoch > 0:#retraining; replay buffer must be filled again
        while replay_buffer.steps_counter < args.min_replay_size:
            agent_episode(0, 0, env, agent, replay_buffer, args, ou_noiser, do_train=False, recovering=True)
    best_success_rate = 0.1
    for epoch in range(start_epoch, args.training_epochs):
        #TODO HGG sample goals for each episode and update hgg sampler to choose goals to explore
        start_time = time.time()
        successes = 0
        for episode in range(args.training_episodes):
            #let the agent train in the environment
            mean_cr1_loss, mean_cr2_loss, mean_act_loss, done = agent_episode(epoch, episode, env, agent,
                                                                              replay_buffer, args, ou_noiser)
            successes += float(done)
            args.csv_logger.finish_episode_log(episode)#this was in agent loop
        if epoch%4 == 0:
            reachable = 0
            for episode in range(100):
                rd = agent_reachability_episode(env, agent, args)
                reachable += float(rd)
            reachable_rate = reachable / 100.
            if reachable_rate >= 0.37:
                print('changing ranges')
                env.env.env.progress_ranges()
            print('reachable rate', reachable_rate)
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

        print(psutil.virtual_memory().available)
        print(psutil.virtual_memory().total * 0.22)
        if psutil.virtual_memory().available <= psutil.virtual_memory().total * 0.22:
            '''print('reached memory threshold saving')
            env.env.env.print_ranges()
            del args.compute_reward
            del args.goal_distance
            del env
            del ou_noiser
            agent.save_train_checkpoint(AGENT_CHECKPOINT_NAME + "_tresh", epoch)
            #replay_buffer.save_checkpoint(RPL_FN + "_tresh", epoch)
            print('achieved goals: {}'.format(args.counter_achieved_goals))
            print('the best: {}'.format(best_success_rate))'''
            make_the_last_call(epoch,env, agent, replay_buffer, args, ou_noiser)
            return


        if epoch % args.test_interval == 0:
            mem = env.env.env.get_ranges()
            env.env.env.set_ranges(low=[1.1, 1.5], high=[1.1, 1.5], rest=[1.1, 1.5],
                           mem_low=[1.1, 1.5], mem_high=[1.1, 1.5])
            successes = 0
            for e in range(args.test_episodes):
                b = TrajectoryBufferForVid()
                done = agent_test_episode(env, agent, b, args)
                if done and (successes == 0 or successes == int(args.test_episodes/2) - 1):
                    trajectory = b.get_trajectory()
                    create_rollout_video(args, trajectory, 'achieved_goal_test_epoch_{}_episode_{}'.format(epoch, e))
                if e == args.test_episodes - 1:
                    trajectory = b.get_trajectory()
                    create_rollout_video(args, trajectory, 'last_test_epoch_{}_episode_{}'.format(epoch, e))
                successes += float(done)
                args.test_csv_logger.finish_episode_log(e)
            env.env.env.reset_ranges(mem)
            '''batch = b.sample_batch(args.batch_size, using_goal_as_field=True)
            criticQ1_loss, criticQ2_loss, actor_loss = agent.evaluate_losses(batch)
            args.test_csv_logger.add_log('critic1_loss', criticQ1_loss)
            args.test_csv_logger.add_log('critic2_loss', criticQ2_loss)
            args.test_csv_logger.add_log('actor_loss', actor_loss)'''
            args.test_csv_logger.finish_epoch_log(epoch)
            success_rate = successes/args.test_episodes
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                agent.save(AGENT_BEST)
    agent.save(AGENT_LAST)


def agent_reachability_episode(env, agent, args):
    obs = env.reset()
    obs = extend_obs(obs, args)
    goal = select_goal(obs, args)
    for step in range(args.env_steps):
        state = select_state(obs, args)
        action = agent.get_action(state, goal)
        next_obs, reward, done, _ = env.step(action)
        next_obs = extend_obs(next_obs, args)
        if done or step == args.env_steps - 1 or not next_obs['gripper_visible'] \
                    or next_obs['object_below_table']:
            break
        obs = next_obs
    return done


def agent_episode(epoch, episode, env, agent, replay_buffer, args, ou_noiser, deterministic_sampling=False,do_train=True, recovering=False):
    obs = env.reset()
    obs = extend_obs(obs, args)
    first_obs = copy(obs)
    ou_noiser.reset()
    ##sample hgg goal for episode
    goal = select_goal(obs, args)
    episode_reward = 0
    cr1_loss, cr2_loss, act_loss = None, None, None
    current = Trajectory(obs)
    for step in range(args.env_steps):
        state = select_state(obs, args)
        if recovering:
            if step <= 0.5*args.env_steps:
                action = agent.get_action(state, goal)
            else:
                action = np.random.uniform(-1,1,4)
        else:
            action = agent.get_action(state, goal)
            if not deterministic_sampling:
                action = ou_noiser.get_action(action, step)
        next_obs, reward, done, _ = env.step(action)
        next_obs = extend_obs(next_obs, args)
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
                if done and not recovering:
                    if args.counter_achieved_goals % 100 == 0:
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
        distance = args.goal_distance(next_obs['achieved_goal'], first_obs['desired_goal'])
        args.csv_logger.add_log('distance_to_goal', distance)
        #
        #args.csv_logger.finish_episode_log(episode)
        if epoch % args.checkpoint_interval == 0:
            agent.save_train_checkpoint(AGENT_CHECKPOINT_NAME, epoch)
        return mean_cr1_loss, mean_cr2_loss, mean_act_loss, done

def agent_test_episode(env, agent, replay_buffer, args):
    obs = env.reset()
    obs = extend_obs(obs, args)

    first_obs = copy(obs)
    ##sample hgg goal for episode
    goal = select_goal(obs, args)
    episode_reward = 0
    current = Trajectory(obs)
    for step in range(args.env_steps):
        state = select_state(obs, args)
        action = agent.get_action(state, goal)
        next_obs, reward, done, _ = env.step(action)
        next_obs = extend_obs(next_obs, args)
        episode_reward += reward
        current.store_step(action, next_obs, reward, done)
        if done or step == args.env_steps - 1:#todo make last obs as done???
            replay_buffer.store_trajectory(current)
            break
        obs = next_obs

    args.test_csv_logger.add_log('episode_reward', episode_reward)
    args.test_csv_logger.add_log('episode_steps', step + 1)
    args.test_csv_logger.add_log('success', float(done))
    distance = args.goal_distance(next_obs['achieved_goal'], first_obs['desired_goal'])
    args.test_csv_logger.add_log('distance_to_goal', distance)
    return done


def select_goal(obs, args):
    if args.goal_type == 'latent':
        return obs['goal_latent']
    elif args.goal_type == 'goal_space':
        return obs['desired_goal']
    elif args.goal_type == 'state':
        return obs['goal_state']
    elif args.goal_type == 'concat':
        raise Exception('observation type not valid')
    else:
        raise Exception('goal obs type not valid')


def select_state(obs, args):
    if args.observation_type == 'latent':
        return obs['state_latent']
    elif args.observation_type == 'real':
        return obs['observation']
    elif args.observation_type == 'concat':
        raise Exception('observation type not valid')
    else:
        raise Exception('observation type not valid')

def setup(args, recover_filename=None):
    env = make_env(args)
    env.env.env.set_ranges(low=[1.1, 1.15], high=[1.45, 1.5], rest=[1.15, 1.45],
                           mem_low=[1.1, 1.15], mem_high=[1.45, 1.5])

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

    state_dim += 0# 16#todo do this dyanmically
    agent = TD3(state_dim=state_dim, action_dim=action_dim, goal_dim=goal_dim,
                max_action=env.action_space.high, args=args)
    if recover_filename is not None:
        agent.load(filename=recover_filename)
    ou_noiser = OUNoise(env.action_space)
    return env, agent, ou_noiser, replay_buffer, state_dim, goal_dim, action_dim


def just_test(env, agent, number_iterations, args):
    successes = 0
    not_successes = 0
    for e in range(number_iterations):
        b = TrajectoryBufferForVid()
        done = agent_test_episode(env, agent, b, args)
        if done and (successes < 20):
            trajectory = b.get_trajectory()
            create_rollout_video(args, trajectory, 'just_test_achieved_goal_test_episode_{}'.format( e))
        if not done and (not_successes < 20):
            trajectory = b.get_trajectory()
            create_rollout_video(args, trajectory, 'just_test_not_achieved_episode_{}'.format(e))
        successes += float(done)
        not_successes += float(not done)
        args.test_csv_logger.finish_episode_log(e)
    args.test_csv_logger.finish_epoch_log(e)
    print('success rate: {}'.format(successes/number_iterations))
    #batch = b.sample_batch(args.batch_size, using_goal_as_field=True)
    #criticQ1_loss, criticQ2_loss, actor_loss = agent.evaluate_losses(batch)
    #args.test_csv_logger.add_log('critic1_loss', criticQ1_loss)
    #args.test_csv_logger.add_log('critic2_loss', criticQ2_loss)
    #args.test_csv_logger.add_log('actor_loss', actor_loss)
    #args.test_csv_logger.finish_epoch_log(epoch)
    #success_rate = successes / args.test_episodes

if __name__ == "__main__":
    '''import sys
    try:
        a = [bytearray(1024*1000) for _ in range(5*1000)]
        b = [bytearray(680*1000) for _ in range(1000)]
    except:
        print("aishh")'''
    args = get_args_and_initialize()
    env, agent, ou_noiser, replay_buffer, state_dim, goal_dim, action_dim = setup(args, recover_filename='td3_dyn_tresh_248')
    args.counter_achieved_goals = 275
    '''env.env.env.set_ranges(low=[1.23, 1.28], high=[1.32, 1.37], rest=[1.28, 1.32],
                           mem_low=[1.1, 1.23], mem_high=[1.37, 1.5])'''
    #replay_buffer.load_checkpoint('rpl_dyn_tresh_84')
    #env.env.env.set_ranges(low=[1.1, 1.5], high=[1.1, 1.5], rest=[1.1, 1.5],
    #                       mem_low=[1.1, 1.5], mem_high=[1.1, 1.5])
    main_loop(start_epoch=249, env=env, agent=agent, replay_buffer=replay_buffer,ou_noiser=ou_noiser, args=args)
    #del replay_buffer
    #just_test(env, agent, 100, args)

