import argparse
import torch
from utils.os_utils import CSV_Logger

def get_args_and_initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='gym env id', type=str, default='FetchPushLabyrinth-v2')
    # parser.add_argument('--graph', help='g-hgg yes or no', type=bool, default=False)
    parser.add_argument('--goal', help='method of goal generation', type=str, default='custom',
                        choices=['vanilla', 'fixobj', 'interval', 'custom'])
    parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=800, help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--checkpoint_interval', type=int, default=50, metavar='N',
                        help='how many epochs a checkpoint is done')
    parser.add_argument('--dirpath', help='path to to directory save/load the model', type=str,
                        default='/home/erick/RL/HGG-extended/HGG-Extended-master/logsdir/')
    parser.add_argument('--datapath', help='file path to dataset', type=str, default=
    '/home/erick/log_extended/01_06_2020_16_29_48_vae_dataset.pkl')
    # Arguments for VAE Architecture
    parser.add_argument('--latent_dim', help='The size from the latent dimension', type=int, default=20)
    parser.add_argument('--img_dim', help='other arguments, as image size, latent_size', type=int, default=84)
    # Arguments for VAE Wrapper
    parser.add_argument('--wrap_with_vae', help='use vae wrapper or not', type=bool, default=True)
    parser.add_argument('--vae_wrap_kwargs', help='other arguments, as image size, latent_size', type=dict,
                        default={'width':84, 'height':84})
    parser.add_argument('--vae_wrap_filename', help='filename for loading weights', type=str, default='trained_last')
    #arguments for TDM training:
    parser.add_argument('--tdm_training_epochs', help='number of training epochs for TDM', type=int, default=300)#500
    parser.add_argument('--tdm_training_episodes', help='number of training episode per epoch', type=int, default=30)
    parser.add_argument('--tdm_env_steps', help='number stops in one epoch', type=int, default=100)
    parser.add_argument('--buffer_size', help='number of samples', type=int, default=25000)#1000000 transition
    parser.add_argument('--min_replay_size', help='replay buffer size before training', type=int, default=3000)
    parser.add_argument('--training_steps', help='how many times tdm agent train in a step', type=int, default=1)
    parser.add_argument('--evaluation_steps', help='how many times evaluate in episode', type=int, default=10)
    parser.add_argument('--just_latent', help='how many times evaluate in episode', type=bool, default=True)
    parser.add_argument('--max_tau', help='The maximum taur for timelilimit/subgoal', type=int, default=25)


    parser.add_argument('--type_experiment', help='', type=str, required=False)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    #decides if use cpu or gpu
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    args.if_cuda_kwargs = kwargs

    if hasattr(args, 'type_experiment'):
        if args.type_experiment == 'tdm_training':
            args.csv_filename = 'tdm_training_log'
            args.csv_logger = CSV_Logger(['actor_tr_loss', 'critic_tr_loss','actor_eval_loss', 'critic_eval_loss',
                                          'episode_reward','distance_to_goal', 'success', 'episode_steps', 'time'],
                                         args, iteration_fieldnames=['epoch', 'episode'])
        elif args.type_experiment == 'tdm_subgoal':
            args.csv_filename = 'tdm_subgoal_log'
            args.csv_logger = CSV_Logger(['episode_reward', 'distance_to_goal',
                                          'success', 'episode_steps', 'latent_distant_sub', 'distance_sub','time'],
                                         args, iteration_fieldnames=['epoch','episode','sub_goal_loop'])

    #seeds every random source
    torch.manual_seed(args.seed)
    return args