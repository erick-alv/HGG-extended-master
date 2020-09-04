import gym
import numpy as np
from torchvision.utils import save_image

from .fixobj import FixedObjectGoalEnv


class IntervalGoalEnv(FixedObjectGoalEnv):
	def __init__(self, args):
		self.img_size = 84
		FixedObjectGoalEnv.__init__(self, args)

	def generate_goal_old(self):
		if hasattr(self, 'target_goal_center'):
			goal = self.target_goal_center + np.random.uniform(-self.target_range, self.target_range, size=3)
			goal[2] = self.height_offset
		else:
			if self.has_object:
				goal = self.initial_gripper_xpos[:3] + self.target_offset
				if self.args.env=='FetchSlide-v1':
					goal[0] += self.target_range*0.5
					goal[1] += np.random.uniform(-self.target_range, self.target_range)*0.5
				else:
					goal[0] += np.random.uniform(-self.target_range, self.target_range)
					goal[1] += np.random.uniform(-self.target_range, self.target_range)
				goal[2] = self.height_offset + int(self.target_in_the_air)*0.45
			else:
				goal = self.initial_gripper_xpos[:3] + np.array([np.random.uniform(-self.target_range, self.target_range), self.target_range, self.target_range])
		return goal.copy()

	def generate_goal(self):
		return self.generate_goal_old()#todo set to one that is far
		'''l = [8,10, 14,27,30,40,41, 42, 43, 46, 59, 63,66]
		idx = np.random.randint(0, len(l))
		#goal = goal_set[np.random.randint(20)]#todo..
		goal = goal_set[l[idx]]
		goal = self.goal_vae.format(goal)
		save_image(goal.cpu().view(-1, 3, self.img_size, self.img_size), 'goal.png')
		x, y = self.goal_vae.encode(goal)
		goal = self.goal_vae.reparameterize(x, y)
		goal = goal.detach().cpu().numpy()
		goal = np.squeeze(goal)
		return goal.copy()'''#TODO..
