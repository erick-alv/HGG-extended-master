from envs import make_env
from utils.os_utils import get_arg_parser
import numpy as np
import time
from datetime import datetime
import pickle
from utils.image_util import rgb_array_to_image

def generate_vae_dataset(args):
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
    for _ in range(10):
        env.render()
    state = env.reset()['observation']#todo use_inter_sampling = True
    action = np.array(env.action_space.sample())
    rgb_array = np.array(env.render(mode='rgb_array', width=84, height=84))

    dataset = {
        'obs': np.zeros(tuple([N]) + rgb_array.shape, dtype=rgb_array.dtype),
        'actions': np.zeros(tuple([N])+ action.shape, dtype=action.dtype),
        'next_obs': np.zeros(tuple([N])+ rgb_array.shape, dtype=rgb_array.dtype),
        'obs_state': np.zeros(tuple([N]) + state.shape, dtype=state.dtype),
        'next_obs_state': np.zeros(tuple([N]) + state.shape, dtype=state.dtype),
    }

    now = time.time()
    for i, ep_steps in enumerate(episode_steps_list):
        env.reset()
        if just_begin_and_end:
            state = env.get_obs()['observation']
            dataset['obs_state'][i, :] = state
            dataset['obs'][i, :] = env.render(mode='rgb_array', width=84, height=84)
            for _ in range(ep_steps):
                if agent:
                    action = agent.step()
                else:
                    action = env.action_space.sample()
                env.step(action)
            #TODO should we leave action to be nothing? or concateante them??
            #dataset['actions'][i, :] = action
            state = env.get_obs()['observation']
            dataset['next_obs_state'][i, :] = state
            dataset['next_obs'][i, :] = env.render(mode='rgb_array', width=84, height=84)
        else:
            for j in range(ep_steps):
                t = i*T+j
                state = env.get_obs()['observation']
                dataset['obs_state'][t, :] = state
                dataset['obs'][t, :] = env.render(mode='rgb_array', width=84, height=84)
                if agent:
                    action = agent.step()
                else:
                    action = env.action_space.sample()
                env.step(action)
                dataset['actions'][t, :] = action
                state = env.get_obs()['observation']
                dataset['next_obs_state'][t, :] = state
                dataset['next_obs'][t, :] = env.render(mode='rgb_array', width=84, height=84)
    print("keys and shapes:")
    for k in dataset.keys():
        print(k, dataset[k].shape)
    filename = args.savefolder_path + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + '_vae_dataset.pkl'
    print(filename)
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    print("done making training data, took:", time.time() - now)

def make_gif_from_images(args):
    with open(args.samples_file_path, 'rb') as f:
        dataset = pickle.load(f)
    N = dataset['next_obs'].shape[0]
    images = [rgb_array_to_image(dataset['next_obs'][i]) for i in range(N)]
    filename = args.savefolder_path + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + '_sample_all.gif'
    images[0].save(filename, save_all=True, append_images=images[1:],
                   optimize=True, loop=1, duration=100)


if __name__=='__main__':
    parser = get_arg_parser()
    parser.add_argument('--env', help='gym env id', type=str, default='FetchPushLabyrinth-v2')
    #parser.add_argument('--graph', help='g-hgg yes or no', type=bool, default=False)
    parser.add_argument('--goal', help='method of goal generation', type=str, default='custom',
                        choices=['vanilla', 'fixobj', 'interval', 'custom'])
    parser.add_argument('--total_samples', help='total number of samples', type=int, default=50000)
    #parser.add_argument('--total_samples', help='total number of samples', type=int, default=)
    parser.add_argument('--interaction_steps', help='total number of samples', type=int, default=15)
    parser.add_argument('--just_begin_and_end', help='if true just first state and end state are taken',
                        type=bool, default=False)
    parser.add_argument('--interaction_agent', help='agent, which will interact with the enviroment',
                        type=str, default='random', choices=['hgg_trained','random'])
    parser.add_argument('--agent_path', help='path to load the corresponding agent', type=str, default='')
    parser.add_argument('--savefolder_path', help='path for storing samples', type=str,
                        default='/home/erick/log_extended/')
    #TODO this argument actually belongs to con_vae
    parser.add_argument('--samples_file_path', help='stored', type=str,
                        default='/home/erick/log_extended/01_06_2020_16_18_26_vae_dataset.pkl')
    args = parser.parse_args()
    generate_vae_dataset(args)
    #make_gif_from_images(args)