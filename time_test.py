import numpy as np
import time
from envs import make_env
from common import get_args, load_vaes
from copy import deepcopy
import pickle
import torch
import tensorflow as tf
from gym.envs.registration import register
from collections import namedtuple
from vae_env_inter import take_env_image, take_objects_image_training, take_image_objects

BufferMock = namedtuple('Buffer', ['counter'])
if __name__ == '__main__':
    # Getting arguments from command line + defaults
    # Set up learning environment including, gym env, ddpg agent, hgg/normal learner, tester
    args = get_args()
    load_vaes(args)
    env = make_env(args)

    args.logger.summary_init(None, None)

    # Progress info
    args.logger.add_item('N')
    args.logger.add_item('Epoch')
    args.logger.add_item('Cycle')
    args.logger.add_item('TimeCost(sec)')


    args.logger.summary_setup()
    counter = 0

    images = []
    env.reset()
    im = take_image_objects(env, 64)
    images.append(im)

    for timestep in range(100):
        action = env.action_space.sample()
        im = take_image_objects(env, 64)
        images.append(im)


    #Testing
    N_confs = [0, 2, 4, 7, 10]

    for n in N_confs:
        print("the current N is {}".format(n))
        for epoch in range(args.epoches):
            for cycle in range(args.cycles):
                args.logger.tabular_clear()
                args.logger.summary_clear()
                start_time = time.time()

                # Log learning progresss
                tester.cycle_summary()
                args.logger.add_record('N', str(n))
                args.logger.add_record('Epoch', str(epoch) + '/' + str(args.epoches))
                args.logger.add_record('Cycle', str(cycle) + '/' + str(args.cycles))
                args.logger.add_record('TimeCost(sec)', time.time() - start_time)

                # Save learning progress to progress.csv file
                args.logger.save_csv()
                args.logger.tabular_show(args.tag)

            self.args.logger.add_record('Success', acc)
            self.args.logger.add_record('MaxDistance', maxDist)
            self.args.logger.add_record('MinDistance', minDist)

            tester.epoch_summary()

    tester.final_summary()