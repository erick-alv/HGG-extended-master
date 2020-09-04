import time
from copy import copy, deepcopy
from algorithm.replay_buffer import Trajectory, ReplayBuffer_Episodic
from utils.transforms_utils import extend_obs, BA
from memory_profiler import profile
import numpy as np
import psutil
from distance_evaluation import create_rollout_video, TrajectoryBufferForVid

AGENT_CHECKPOINT_NAME = 'td3_dyn'
AGENT_LAST = 'td3_dyn_last'
AGENT_BEST = 'td3_dyn_best'
RPL_FN = 'rpl_dyn'
CSV_VID_PATTERN = 'td3_dyn_epoch_{}_it_{}'#'dist_no_obstacle_epoch_{}_it_{}'

@profile()
def make_the_last_call(last_epoch, env, agent, replay_buffer, args, ou_noiser=None):
    #TODO HGG sample goals for each episode and update hgg sampler to choose goals to explore
    start_time = time.time()
    successes = 0
    for episode in range(10):
        #let the agent train in the environment
        mean_cr1_loss, mean_cr2_loss, mean_act_loss, done = agent_episode_last(last_epoch, episode, env, agent,
                                                                          replay_buffer, args, ou_noiser)
        successes += float(done)
        args.csv_logger.finish_episode_log(episode)#this was in agent loop
    reachable = 0
    for episode in range(10):
        rd = agent_reachability_episode_last(env, agent, args)
        reachable += float(rd)
    reachable_rate = reachable / 10.
    print('changing ranges')
    env.env.env.progress_ranges()
    print('reachable rate', reachable_rate)
    time_needed = time.time() - start_time
    print('finished epoch ', last_epoch)
    print('last actor loss ', mean_act_loss)
    print('last_critic1 loss ', mean_cr1_loss)
    print('last_critic2 loss ', mean_cr2_loss)
    print('success rate', successes/args.training_episodes)

    print('time needed ', time_needed)
    print('______________________________')
    args.csv_logger.add_log('time', time_needed)
    args.csv_logger.finish_epoch_log(last_epoch)

    #testing
    mem = env.env.env.get_ranges()
    env.env.env.set_ranges(low=[1.1, 1.5], high=[1.1, 1.5], rest=[1.1, 1.5],
                   mem_low=[1.1, 1.5], mem_high=[1.1, 1.5])
    successes = 0
    for e in range(args.test_episodes):
        b = TrajectoryBufferForVid()
        done = agent_test_episode_last(env, agent, b, args)
        if done and (successes == 0 or successes == int(args.test_episodes/2) - 1):
            trajectory = b.get_trajectory()
            create_rollout_video(args, trajectory, 'achieved_goal_test_epoch_{}_episode_{}'.format(last_epoch, e))
        if e == args.test_episodes - 1:
            trajectory = b.get_trajectory()
            create_rollout_video(args, trajectory, 'last_test_epoch_{}_episode_{}'.format(last_epoch, e))
        successes += float(done)
        args.test_csv_logger.finish_episode_log(e)
    env.env.env.reset_ranges(mem)
    args.test_csv_logger.finish_epoch_log(last_epoch)
    success_rate = successes/args.test_episodes
    agent.save(AGENT_LAST)

    print('reached memory threshold saving')
    env.env.env.print_ranges()
    del args.compute_reward
    del args.goal_distance
    del env
    del ou_noiser
    agent.save_train_checkpoint(AGENT_CHECKPOINT_NAME + "_tresh", last_epoch)
    #replay_buffer.save_checkpoint(RPL_FN + "_tresh", epoch)
    print('achieved goals: {}'.format(args.counter_achieved_goals))
    return

@profile()
def agent_reachability_episode_last(env, agent, args):
    obs = env.reset()
    obs = extend_obs(obs, args)
    goal = select_goal_last(obs, args)
    for step in range(args.env_steps):
        state = select_state_last(obs, args)
        action = agent.get_action(state, goal)
        next_obs, reward, done, _ = env.step(action)
        next_obs = extend_obs(next_obs, args)
        if done or step == args.env_steps - 1 or not next_obs['gripper_visible'] \
                    or next_obs['object_below_table']:
            break
        obs = next_obs
    return done

@profile()
def agent_episode_last(epoch, episode, env, agent, replay_buffer, args, ou_noiser, deterministic_sampling=False,do_train=True, recovering=False):
    obs = env.reset()
    obs = extend_obs(obs, args)
    first_obs = copy(obs)
    ou_noiser.reset()
    ##sample hgg goal for episode
    goal = select_goal_last(obs, args)
    episode_reward = 0
    cr1_loss, cr2_loss, act_loss = None, None, None
    current = Trajectory(obs)
    for step in range(args.env_steps):
        state = select_state_last(obs, args)
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

@profile()
def agent_test_episode_last(env, agent, replay_buffer, args):
    obs = env.reset()
    obs = extend_obs(obs, args)

    first_obs = copy(obs)
    ##sample hgg goal for episode
    goal = select_goal_last(obs, args)
    episode_reward = 0
    current = Trajectory(obs)
    for step in range(args.env_steps):
        state = select_state_last(obs, args)
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


def select_goal_last(obs, args):
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


def select_state_last(obs, args):
    if args.observation_type == 'latent':
        return obs['state_latent']
    elif args.observation_type == 'real':
        return obs['observation']
    elif args.observation_type == 'concat':
        raise Exception('observation type not valid')
    else:
        raise Exception('observation type not valid')