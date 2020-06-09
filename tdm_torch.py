from custom_start import get_args_and_initialize
import torch
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#more than a model this learns a q function based on goal_conditioned
class TDM:
    def __init__(self, act_cr_model, replay_buffer, env, args):
        self.act_cr_model = act_cr_model
        self.replay_buffer = replay_buffer
        self.env = env
        self.args = args

    def training_loop(self, start_epoch):
        for epoch in range(start_epoch, args.epochs + 1):
            #self._start_epoch(epoch)
            observation = self.env.reset()
            trajectory = Trajectory(start_tr=observation)
            for step in range(args.num_env_steps_per_epoch):
                action = self.get_action_exploration(observation)
                #observation = self._take_step_in_env(observation)
                reward, a,b, next_observation = self.env.step(action)
                trajectory.store(reward, observation, next_observation, b)
                if final or max_steps:
                    replay_buffer.store(transition)

                if self.replay_buffer.size() > args.min_replay_size_for_training:#30000:
                    for training_step in range(args.training_steps):
                        batch = self.replay_buffer.get_batch_relabeled(M)
                        self.act_cr_model.train(epoch, st, training_step ,batch)
                    #todo DO evaluation

                    #evaluation
                    # end_epoch and move post_epoch after eval
                    #self._post_epoch(epoch)
                    #self._try_to_eval(epoch)
                    #gt.stamp('eval')
                    #self._end_epoch()

                if final:
                    break
                observation = next_observation



def setup_tdm(args, recover_filename=None):
    pass

def start_training(args):
    model, trainer, _, _ = setup_tdm(args)
    training_loop(args, 1, model, trainer)

def resume_training(args, epoch):
    recover_filename = 'trained_weights_'+str(epoch)
    model, trainer, _, _ = setup_tdm(args, recover_filename=recover_filename)
    training_loop(args, epoch, model, trainer)

def training_loop(args, start_epoch, model, trainer):
    for epoch in range(start_epoch, args.epochs + 1):
        trainer.train(epoch)
        trainer.test(epoch)
        if epoch % 100 == 0 or epoch == args.epochs:
            with torch.no_grad():
                sample = torch.randn(64, model.latent_dim).to(args.device)
                sample = model.decode(sample).cpu()
                save_image(sample.view(64, model.channels_dim, model.x_y_dim, model.x_y_dim),
                           args.dirpath + 'results/sample_' + str(epoch) + '.png')
    save_vae(args, 'trained_last', vae=model, trainer=trainer)

if __name__ == "__main__":
    args = get_args_and_initialize()    #start_training(args)
    resume_training(args, 400)