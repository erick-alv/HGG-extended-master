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