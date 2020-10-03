import gym
import numpy as np
from .vanilla import VanillaGoalEnv


class FixedObjectGoalEnv(VanillaGoalEnv):
	def __init__(self, args):
		VanillaGoalEnv.__init__(self, args)
		self.env.reset()
		self.fixed_obj = True

	def reset(self):
		self.sim.set_state(self.initial_state)

		if self.has_object:
			if hasattr(self, 'object_center'):
				object_xpos = self.object_center[:2] \
							  + np.random.uniform(-self.obj_range, self.obj_range, size=2)
				object_qpos = self.sim.data.get_joint_qpos('object0:joint')
				assert object_qpos.shape == (7,)
				object_qpos[:2] = object_xpos
				#object_qpos[3:] = np.array([1., 0., 0., 0.])
			else:
				object_xpos = self.initial_gripper_xpos[:2].copy()
				random_offset = np.random.uniform(-1,1)*self.obj_range*self.args.init_offset
				if self.args.env=='FetchSlide-v1': object_xpos -= np.array([self.obj_range*0.5, random_offset])
				else: object_xpos -= np.array([random_offset, self.obj_range])
				object_qpos = self.sim.data.get_joint_qpos('object0:joint')
				assert object_qpos.shape == (7,)
				object_qpos[:2] = object_xpos
			self.sim.data.set_joint_qpos('object0:joint', object_qpos)

		self.sim.forward()
		self.goal = self.generate_goal()
		self.reset_ep()#must be done here so images are correct
		self.last_obs = (self.get_obs()).copy()
		return self.get_obs()

	def generate_goal(self):
		return self.env.env._sample_goal()