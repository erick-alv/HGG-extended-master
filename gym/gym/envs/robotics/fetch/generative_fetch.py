import os
from gym import utils
from gym.envs.robotics import fetch_env
import numpy as np
from scipy.spatial.transform import Rotation as Rot


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'generative_fetch.xml')

class FetchGenerativeEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        self.further = False

        # TODO: configure adaption parameters
        self.adapt_dict = dict()
        self.adapt_dict["field"] = [1.3, 0.75, 0.6, 0.25, 0.25, 0.2]


        self.target_goal_center = np.array([1.3, 0.57, 0.425])
        self.object_center = np.array([1.3, 0.93, 0.425])


        initial_qpos = {
            'robot0:slide0': 0.0,
            'robot0:slide1': 0.0,
            'robot0:slide2': 0.0,
            #'object0:joint': [1.25, 0.63, 0.4, 1., 0., 0., 0.],  # origin 0.53#this might be the reason for the funny (behaviour, we should wait til sim ends??#
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.02, target_range=0.02, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    # RobotEnv methods
    # ----------------------------

    def _sample_goal(self):
        goal = self.target_goal_center + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        goal[2] = self.height_offset
        return goal.copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        object_xpos = self.object_center[:2] + self.np_random.uniform(-self.obj_range,
                                                                      self.obj_range, size=2)
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        object_qpos[3:] = np.array([1., 0.,0.,0.])
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True


    def _rotate(self, names_list, rot_x, rot_y, rot_z):
        #rot_z and rot_x changed to be congrunet wit angles used in space
        r = Rot.from_rotvec(np.pi/180 * np.array([ rot_z, rot_y, rot_x]))
        q = r.as_quat()
        for name in names_list:
            id = self.sim.model.body_name2id(name)
            self.sim.model.body_quat[id] = q.copy()

    def _change_color(self, names_list, r, g, b):
        for name in names_list:
            id = np.where(self.sim.model.geom_bodyid == self.sim.model.body_name2id(name))
            id = id[0].item()
            self.sim.model.geom_rgba[id] = np.array([r,g,b,1.])



'''if __name__ == '__main__':
    import time
    from utils.image_util import rgb_array_to_image
    import cv2
    e = FetchPushMovingObstacleEnv()
    im_size = 500
    ims = []
    for episode in range(5):
        ob = e.reset()
        for step in range(100):
            array = e.render(mode='rgb_array')
            image = rgb_array_to_image(array)
            ims.append(image)
            a = e.action_space.sample()
            e.step(a)
            obs = e._get_obs()
    e.close()
    out = cv2.VideoWriter('/home/erick/RL/HGG-extended/HGG-Extended-master/vid.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          10, (im_size, im_size))
    for i in range(len(ims)):
        out.write(np.array(ims[i]))
    out.release()'''