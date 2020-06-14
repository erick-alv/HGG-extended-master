import argparse
import torch

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
    parser.add_argument('--checkpoint_interval', type=int, default=100, metavar='N',
                        help='how many epochs a checkpoint is done')
    parser.add_argument('--dirpath', help='path to to directory save/load the model', type=str,
                        default='/home/erick/log_extended/')
    parser.add_argument('--datapath', help='file path to dataset', type=str, default=
    '/home/erick/log_extended/01_06_2020_16_29_48_vae_dataset.pkl')
    ##Arguments for VAE_Wrapper
    parser.add_argument('--wrap_with_vae', help='use vae wrapper or not', type=bool, default=True)
    parser.add_argument('--vae_wrap_kwargs', help='other arguments, as image size, latent_size', type=dict,
                        default={'width':84, 'height':84})
    parser.add_argument('--vae_wrap_filename', help='filename for loading weights', type=str, default='trained_last')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(torch.cuda.is_available())

    #decides if use cpu or gpu
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    args.if_cuda_kwargs = kwargs

    #seeds every random source
    torch.manual_seed(args.seed)
    return args