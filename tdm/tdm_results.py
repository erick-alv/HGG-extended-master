from tdm.tdm_torch import setup_tdm
from custom_start import get_args_and_initialize
from copy import copy
from utils.image_util import store_image_array_at, make_video
if __name__=='__main__':
    args = get_args_and_initialize()
    _, td3_actor_critic, _, _, env = setup_tdm(args, recover_filename='td3_tr_last')
    td3_actor_critic.critic.eval()
    td3_actor_critic.actor.eval()

    obs = env.reset()
    first_obs = copy(obs)
    goal = obs['goal_latent']
    goal_im = obs['goal_image']
    episode_reward = 0

    for step in range(args.tdm_env_steps):
        rem_steps = args.max_tau - (step % (args.max_tau + 1))
        state = obs['state_latent']
        action = td3_actor_critic.get_action(state, goal, rem_steps)[0]
        q_val = td3_actor_critic.get_Q_val(state, action, goal, rem_steps)[0][0]
        store_image_array_at(obs['state_image'], args.dirpath+'images_for_video/', 'frame_{}'.format(step),
                             text_append='V:\n {}'.format(q_val))
        next_obs, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            break
        obs = next_obs
    action = td3_actor_critic.get_action(goal, goal, 10)[0]
    q_val = td3_actor_critic.get_Q_val(goal, action, goal, 10)[0][0]
    store_image_array_at(goal_im, args.dirpath + 'images_for_video/', 'frame_{}'.format(step),
                         text_append='V_goal:\n {}'.format(q_val))
    make_video('/home/erick/RL/HGG-extended/HGG-Extended-master/logsdir/images_for_video/', '.png')
