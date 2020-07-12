from envs import make_env
from utils.os_utils import get_arg_parser
import numpy as np
import time
from datetime import datetime
import pickle
from utils.image_util import rgb_array_to_image

def generate_vae_dataset(args, save_file_name):
    N = args.total_samples
    T = args.interaction_steps
    just_begin_and_end = args.just_begin_and_end
    if just_begin_and_end:
        episode_steps_list = [T]*N
    else:
        n = N // T
        episode_steps_list = [T]*n
        last_steps = N - T*n
        if last_steps > 0:
            episode_steps_list += [last_steps]

    if args.interaction_agent == 'hgg_trained':
        path = args.agent_path
        #TODO load the agent with exception handling
        #try:
        #    agent = load ....
    elif args.interaction_agent == 'random':
        agent = None

    #Prepare dict
    #just to know the types and shapes
    env = make_env(args)
    obs = env.reset()
    state = obs['observation']#todo use_inter_sampling = True
    achieved_goal = obs['achieved_goal']
    action = np.array(env.action_space.sample())
    rgb_array = np.array(env.render(mode='rgb_array', width=args.img_dim, height=args.img_dim))

    dataset = {
        'state': np.zeros(tuple([N]) + state.shape, dtype=state.dtype),
        'state_image': np.zeros(tuple([N]) + rgb_array.shape, dtype=rgb_array.dtype),
        'achieved_goal': np.zeros(tuple([N]) + tuple([3]), dtype= achieved_goal.dtype),
        'actions': np.zeros(tuple([N])+ action.shape, dtype=action.dtype),
        'next_state': np.zeros(tuple([N]) + state.shape, dtype=state.dtype),
        'next_state_image': np.zeros(tuple([N])+ rgb_array.shape, dtype=rgb_array.dtype),
        'next_achieved_goal': np.zeros(tuple([N]) + tuple([3]), dtype= achieved_goal.dtype),
        'goal_state': np.zeros(tuple([N]) + state.shape, dtype=state.dtype),
        'goal_image': np.zeros(tuple([N]) + rgb_array.shape, dtype=rgb_array.dtype),
        'goal_pos': np.zeros(tuple([N]) + tuple([3]), dtype=achieved_goal.dtype)
    }

    now = time.time()
    for i, ep_steps in enumerate(episode_steps_list):
        env.reset()
        if just_begin_and_end:

            dataset['state'][i, :] = obs['observation']
            dataset['achieved_goal'][i, :] = obs['achieved_goal']
            dataset['state_image'][i, :] = env.render(mode='rgb_array', width=args.img_dim, height=args.img_dim)
            for _ in range(ep_steps):
                if agent:
                    action = agent.step()
                else:
                    action = env.action_space.sample(np.zeros(tuple([N])+ rgb_array.shape, dtype=rgb_array.dtype),)
                env.step(action)

            dataset['next_state'][i, :] = obs['observation']
            dataset['next_state_image'][i, :] = env.render(mode='rgb_array', width=args.img_dim, height=args.img_dim)
            dataset['next_achieved_goal'][i, :] = obs['achieved_goal']
            dataset['goal_image'][i, :] = env.env.env.goal_image
            dataset['goal_pos'][i, :] = env.env.env.goal_pos
            dataset['goal_state'][i, :] = env.env.env.goal_state
        else:
            for j in range(ep_steps):
                t = i*T+j
                obs = env.get_obs()
                dataset['state'][t, :] = obs['observation']
                dataset['achieved_goal'][t, :] = obs['achieved_goal']
                dataset['state_image'][t, :] = env.render(mode='rgb_array',
                                                          width=args.img_dim, height=args.img_dim)
                if agent:
                    action = agent.step()
                else:
                    action = env.action_space.sample()
                env.step(action)
                dataset['actions'][t, :] = action
                obs = env.get_obs()
                dataset['next_state'][t, :] = obs['observation']
                dataset['next_achieved_goal'][t, :] = obs['achieved_goal']
                dataset['next_state_image'][t, :] = env.render(mode='rgb_array',
                                                               width=args.img_dim, height=args.img_dim)

                dataset['goal_image'][t, :] = env.env.env.goal_image
                dataset['goal_pos'][t, :] = env.env.env.goal_pos
                dataset['goal_state'][t, :] = env.env.env.goal_state
    print("keys and shapes:")
    for k in dataset.keys():
        print(k, dataset[k].shape)
    filename = args.dirpath + 'training_images/'+  datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + save_file_name
    print(filename)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    print("done making training data, took:", time.time() - now)

'''def make_gif_from_images(args):
    with open(args.samples_file_path, 'rb') as f:
        dataset = pickle.load(f)
    N = dataset['next_obs'].shape[0]
    images = [rgb_array_to_image(dataset['next_obs'][i]) for i in range(N)]
    filename = args.savefolder_path + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + '_sample_all.gif'
    images[0].save(filename, save_all=True, append_images=images[1:],
                   optimize=True, loop=1, duration=100)'''


if __name__=='__main__':
    parser = get_arg_parser()
    parser.add_argument('--env', help='gym env id', type=str, default='FetchPushLabyrinth-v3')
    parser.add_argument('--goal', help='method of goal generation', type=str, default='custom',
                        choices=['vanilla', 'fixobj', 'interval', 'custom'])
    parser.add_argument('--total_samples', help='total number of samples', type=int, default=50000)
    #parser.add_argument('--total_samples', help='total number of samples', type=int, default=)
    parser.add_argument('--interaction_steps', help='total number of samples', type=int, default=20)
    parser.add_argument('--img_dim', help='other arguments, as image size, latent_size', type=int, default=64)
    parser.add_argument('--just_begin_and_end', help='if true just first state and end state are taken',
                        type=bool, default=False)
    parser.add_argument('--interaction_agent', help='agent, which will interact with the enviroment',
                        type=str, default='random', choices=['hgg_trained','random'])
    parser.add_argument('--agent_path', help='path to load the corresponding agent', type=str, default='')
    parser.add_argument('--dirpath', help='path to to directory save/load the model', type=str,
                        default='/home/erick/RL/HGG-extended/HGG-Extended-master/logsdir/')
    args = parser.parse_args()
    generate_vae_dataset(args, '_vae_dataset_no_obstacles.pkl')
    #make_gif_from_images(args)