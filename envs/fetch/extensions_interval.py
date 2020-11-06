from .interval import IntervalGoalEnv

class IntervalExt(IntervalGoalEnv):
    def __init__(self, args):
        IntervalGoalEnv.__init__(self, args)

    def get_obs(self):
        obs = super(IntervalExt, self).get_obs()
        sim = self.env.env.sim
        exists_collision = False
        #todo generalize this for other environments
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            if (contact.geom1 == 23 and contact.geom2 == 24) or (contact.geom1 == 24 and contact.geom2 == 23):
                exists_collision = True

        obs['collision_check'] = exists_collision
        return obs

    def get_obs(self):
        if (hasattr(self.args, 'transform_dense') and self.args.transform_dense) or \
                (hasattr(self.args, 'vae_dist_help') and self.args.vae_dist_help):
            obs = self.env.env._get_obs()
            obs['desired_goal_latent'] = self.desired_goal_latent.copy()
            obs['achieved_goal_latent'] = self.achieved_goal_latent.copy()
            # obs['achieved_goal_image'] = self.achieved_goal_image.copy()
            obs['obstacle_latent'] = self.obstacle_latent.copy()
            obs['obstacle_size_latent'] = self.obstacle_size_latent.copy()
            return obs
        else:
            return self.env.env._get_obs()

    def step(self, action):#just under step we can assure that a transitions in time has ocuured therefore here increment counter
        # imaginary infinity horizon (without done signal)
        obs, reward, done, info = self.env.step(action)
        if (hasattr(self.args, 'transform_dense') and self.args.transform_dense) or \
                (hasattr(self.args, 'vae_dist_help') and self.args.vae_dist_help):
            if self.args.vae_type == 'monet' or self.args.vae_type == 'space' or self.args.vae_type == 'bbox':
                achieved_image = take_image_objects(self, self.args.img_size)
                if self.args.vae_type == 'space' or self.args.vae_type == 'bbox':
                    lg, lg_s, lo, lo_s = latents_from_images(np.array([achieved_image]), self.args)
                else:
                    lg, lo, lo_s = latents_from_images(np.array([achieved_image]), self.args)
                self.achieved_goal_latent = lg[0].copy()
                self.obstacle_latent = lo[0].copy()
                self.obstacle_size_latent = lo_s[0].copy()
            else:
                achieved_goal_image = take_goal_image(self, self.args.img_size)
                latents_goal = goal_latent_from_images(np.array([achieved_goal_image]), self.args)
                self.achieved_goal_latent = latents_goal[0].copy()

                obstacle_image = take_obstacle_image(self, self.args.img_size)
                latents_obstacle, latents_o_size = obstacle_latent_from_images(np.array([obstacle_image]), self.args)
                self.obstacle_latent = latents_obstacle[0].copy()
                self.obstacle_size_latent = latents_o_size[0].copy()
            # self.achieved_goal_image = achieved_goal_image.copy()

            if hasattr(self.args, 'transform_dense') and self.args.transform_dense:
                reward = -self.args.compute_reward_dense(self.obstacle_latent.copy(), self.obstacle_size_latent.copy(),
                                                         self.achieved_goal_latent.copy(),
                                                         self.desired_goal_latent.copy(),
                                                         range_x=[-1., 1.], range_y=[-1., 1.])
                if not np.isscalar(reward):
                    reward = reward[0]
            info = self.process_info(obs, reward, info)

        else:
            info = self.process_info(obs, reward, info)
            reward = self.compute_reward((obs['achieved_goal'], self.last_obs['achieved_goal']), obs['desired_goal'])
        obs = self.get_obs()
        self.last_obs = obs.copy()
        return obs, reward, False, info

    def reset(self):
        self.env.reset()
        self.reset_ep()
        self.last_obs = self.get_obs().copy()
        return self.last_obs.copy()