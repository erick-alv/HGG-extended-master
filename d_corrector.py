import torch
import numpy as np
from vae_env_inter import transform_image_to_latent_batch_torch
from torch import nn, optim
from j_vae.train_vae_sb import load_Vae as load_Vae_SB
from j_vae.train_vae import load_Vae
from j_vae.space_visualizer import visualization_grid_points
from envs import make_env
from torch.nn import functional as F
from j_vae.latent_space_transformations import torch_goal_transformation
from j_vae.train_vae_sb import loss_function
class Corrector:
    def __init__(self, args):
        self.env = make_env(args)
        self.previous_list = None
        self.optimizer = optim.Adam(args.vae_model_goal.parameters(), lr=1e-4)
        self.batch_size = args.corrector_batch_size
        self.corrector_epochs = args.corrector_epochs
        self.calls = 0
        self.surpass_counter = 0

    def correct(self, goal_list, goal_list_z, real_goals, real_goals_z, rollout_ims, rollout_rws, args):
        self.calls += 1
        dists_between_goals = np.array(real_goals) - np.array(goal_list)
        dists_between_goals = np.linalg.norm(dists_between_goals, axis=1)
        tr_too_far = 0.5
        too_far = dists_between_goals[dists_between_goals>=tr_too_far]
        prc = len(too_far) / float(len(dists_between_goals))
        tr_prc = 0.6
        if prc >= tr_prc:
            self.surpass_counter +=1
        else:
            self.surpass_counter = 0
        if self.surpass_counter >= 5:
            able_to_correct = True
            self.surpass_counter = 0
        else:
            able_to_correct = False


        if able_to_correct:
            torch.cuda.empty_cache()

            rollout_ims = np.array(rollout_ims)
            rollout_rws = np.array(rollout_rws)
            rollouts_to_use = []
            goals_to_use = []
            rollout_step_length = []
            dists_to_goal = []

            for i, r_ims in enumerate(rollout_ims):
                '''idx_array = np.where(rollout_rws[i] == 1.0)[0]
                if len(idx_array>0):
                    t = idx_array[0]
                    rollouts_to_use.append(r_ims[:t+1])
                    rollout_step_length.append(t+1)'''
                rollouts_to_use.append(r_ims)
                rollout_step_length.append(len(r_ims))
                goals_to_use.append(r_ims[-1])
                dists_to_goal.append(np.flip(np.arange(len(r_ims))))
            rollouts_to_use = np.array(rollouts_to_use)
            rollout_step_length = np.array(rollout_step_length)
            goals_to_use = np.array(goals_to_use)
            dists_to_goal = np.array(dists_to_goal)
            if len(rollouts_to_use) > 0:
                for e in range(self.corrector_epochs):
                    self._correct_goals(rollouts_to_use, rollout_step_length, goals_to_use, dists_to_goal, args)
                    if e == 0 or e == self.corrector_epochs-1:
                        visualization_grid_points(self.env, args.vae_model_goal, 0.045, args.img_size, 7, 'goal',
                                                  ind_1=args.goal_ind_1, ind_2=args.goal_ind_2, use_d=True,
                                                  fig_file_name='{}space_call{}_e{}.png'.format(args.logger.my_log_dir,
                                                                                                self.calls, e))
            args.vae_model_goal.eval()
            torch.cuda.empty_cache()

    def _correct_goals(self, rollout_ims, rollout_step_length, goals_to_use, dists_to_goal, args):
        args.vae_model_goal.train()
        for i, rollout in enumerate(rollout_ims):
            data_set = np.split(rollout, rollout_step_length[i] / self.batch_size)
            dists_to_goal_set = np.split(dists_to_goal[i], rollout_step_length[i] / self.batch_size)
            for batch_idx, data in enumerate(data_set):
                self._correct_representation(data, goals_to_use[i], dists_to_goal_set[batch_idx], args)

    def _correct_representation(self, rollout_ims, goal_im, step_dists, args):#[r1, r2] = [[step_1_im, step2_im, ..., step_last_im], [step_1_im, step2_im, ..., step_last_im]]
        all_ims = rollout_ims.copy()
        all_ims = torch.from_numpy(all_ims).float().to(args.device)
        all_ims /= 255
        all_ims = all_ims.permute([0, 3, 1, 2])
        all_mu_torch, all_logvar_torch = args.vae_model_goal.encode(all_ims)
        all_mu_torch.detach()
        all_logvar_torch.detach()

        g_im = np.expand_dims(goal_im.copy(), axis=0)
        g_im = torch.from_numpy(g_im).float().to(args.device)
        g_im /= 255
        g_im = g_im.permute([0, 3, 1, 2])
        self.optimizer.zero_grad()

        goal_rec, goal_mu, goal_logvar = args.vae_model_goal(g_im)
        vae_loss = loss_function(goal_rec, g_im, goal_mu, goal_logvar, args.corrector_beta)

        st_mu = torch.cat([all_mu_torch[:, args.goal_ind_1].unsqueeze(axis=1),
                           all_mu_torch[:, args.goal_ind_2].unsqueeze(axis=1)],
                          axis=1)
        g_mu = torch.cat([goal_mu[:, args.goal_ind_1].unsqueeze(axis=1), goal_mu[:, args.goal_ind_2].unsqueeze(axis=1)],
                         axis=1)
        torch_step_dist = torch.from_numpy(step_dists.copy()).float().to(args.device)

        o_loss = self._dynamic_loss(st_mu, g_mu, torch_step_dist, args)
        #o_loss = self._temporal_coherence_loss(st_mu, args)
        loss = vae_loss + o_loss
        loss.backward()
        self.optimizer.step()

    def _dynamic_loss(self, rollout_latents, goal_latent, step_dists, args):
        st = rollout_latents[:-1]
        st1 = rollout_latents[1:]
        delta_t = st1 - st
        mean_dist = torch.norm(delta_t, dim=1).mean()
        step_dists = mean_dist*step_dists
        real_dists = goal_latent - rollout_latents
        real_dists = torch.norm(real_dists, dim=1)
        dyn_loss = 0.5 * nn.MSELoss()(real_dists, step_dists)
        return dyn_loss

    def _temporal_coherence_loss(self, all_mu, args):#todo apply to evrythin
        st = all_mu[:-1]
        st1 = all_mu[1:]
        delta_t = st1 - st
        temp_loss = torch.norm(delta_t, dim=1)
        temp_loss = temp_loss.pow(2).mean()
        return temp_loss


def add_distances(distances, threshold_a, threshold_b):
    pass

'''if __name__ == '__main__':
    #temporal_coherence_loss(random_rollouts)
    #dynamic_loss(random_rollouts)
    a = np.random.rand(5,84,84,3)
    b = np.random.rand(7,84,84,3)
    c = np.random.rand(2,84,84,3)
    rollouts_ims = np.array([a, b, c])
    rollout_step_length = [len(r) for r in rollouts_ims]
    random_rollouts = torch.rand((14, 2))
    random_rollouts = list(torch.split(random_rollouts, rollout_step_length))'''
