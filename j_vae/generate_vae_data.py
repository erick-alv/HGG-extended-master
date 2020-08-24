import numpy as np
from common import get_args, experiment_setup
import time
from PIL import Image
from envs import make_env

if __name__ == '__main__':
    # Getting arguments from command line + defaults
    # Set up learning environment including, gym env, ddpg agent, hgg/normal learner, tester
    args = get_args()
    env = make_env(args)

    # Generate VAE-dataset
    # Train VAE
    # Show mocap settings in robot_env.py!
    count = 1280*15
    img_size = 84
    train_data = np.empty([count, img_size, img_size, 3])
    print('Generating ', count, ' images for VAE-training...')
    for i in range(count):
        env.env.env._generate_state()
        rgb_array = np.array(env.render(mode='rgb_array', width=img_size, height=img_size))
        train_data[i] = rgb_array
        #img = Image.fromarray(rgb_array).show()
        #time.sleep(1)

    np.save('../data/FetchPush/obs_set.npy', train_data)
    print('Finished generating dataset!')
