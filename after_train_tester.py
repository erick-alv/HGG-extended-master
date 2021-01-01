import numpy as np
import time
from common import get_args, experiment_setup_test
from copy import deepcopy
import pickle
import torch
import tensorflow as tf
from gym.envs.registration import register
from collections import namedtuple


BufferMock = namedtuple('Buffer', ['counter'])
if __name__ == '__main__':
    # Getting arguments from command line + defaults
    # Set up learning environment including, gym env, ddpg agent, hgg/normal learner, tester
    args = get_args(do_just_test=True)
    env, agent, tester = experiment_setup_test(args)
    args.logger.summary_init(None, None)
    args.buffer = BufferMock(counter=0)

    '''#activate test setup if exists
    if hasattr(tester.env.env.env,'test_setup'):
        for e in tester.env_List:
            e.env.env.test_setup()'''

    # Progress info
    args.logger.add_item('N')
    args.logger.add_item('Epoch')
    args.logger.add_item('Cycle')
    args.logger.add_item('TimeCost(sec)')


    # Test info
    for key in tester.info:
        args.logger.add_item(key, 'scalar')

    args.logger.summary_setup()
    counter = 0

    #Testing
    N_confs = [0, 2, 4, 7, 10]
    for n in N_confs:
        print("the current N is {}".format(n))
        tester.coll_tol = n
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

            tester.epoch_summary()

    tester.final_summary()
