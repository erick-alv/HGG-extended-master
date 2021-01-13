import gym
import numpy as np
from torchvision.utils import save_image

from .fixobj import FixedObjectGoalEnv


class IntervalGoalEnv(FixedObjectGoalEnv):
	def __init__(self, args):
		self.img_size = args.img_size
		FixedObjectGoalEnv.__init__(self, args)

	def generate_goal(self):
		if self.target_goal_center is not None:
			ndim = self.target_goal_center.ndim
			if ndim > 1:
				ind = np.random.randint(len(self.target_goal_center))
				goal_center = self.target_goal_center[ind]
			else:
				goal_center = self.target_goal_center
			if isinstance(self.target_range, np.ndarray):
				if self.target_range.size == 2:
					range_to_use = np.concatenate([self.target_range, np.zeros(shape=1)])
				elif self.target_range.size == 3:
					range_to_use = self.target_range, np.zeros(shape=1)
				offset = np.random.uniform(-range_to_use, range_to_use)
			else:
				offset = np.random.uniform(-self.target_range, self.target_range, size=3)
			goal = goal_center + offset
			goal[2] = goal_center[2]
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
