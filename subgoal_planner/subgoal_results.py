from utils.image_util import stack_images_column_2, stack_images_row, store_image_array_at, make_video
from os import listdir
from os.path import isfile, join
from custom_start import get_args_and_initialize
from PIL import Image
import numpy as np

def group_images_into_frames(args):
    pr_load =args.dirpath+'images_for_video/'
    pr_save = args.dirpath+'images_for_video/grouped/'
    images_files = [f for f in listdir(pr_load)
                    if isfile(join(args.dirpath+'images_for_video/', f))]
    subgoals_images_files = [f for f in images_files if 'subgoal' in f]
    goals_files = [f for f in images_files if 'interval_3.png' in f]
    goals_files.sort()
    gs = [np.asarray(Image.open(pr_load + f))
              for f in goals_files]
    g = stack_images_row(gs)
    for interval in range(4):
        subgoals_interval = [f for f in subgoals_images_files if 'interval_{}.png'.format(interval) in f]
        subgoals_interval.sort()
        sgs = [np.asarray(Image.open(pr_load + f))
              for f in subgoals_interval]
        sg = stack_images_row(sgs)
        for step in range(25):
            step_interval_frames_files = [f for f in images_files
                                          if 'frame' in f and 'interval_{}_step_{}.png'.format(interval, step) in f]
            step_interval_frames_files.sort()
            ims = [np.asarray(Image.open(pr_load + f))
                   for f in step_interval_frames_files]
            im = stack_images_row(ims)

            frame = stack_images_column_2([im, sg, g])
            store_image_array_at(frame, pr_save, 'frame_{}'.format(25*interval+step))

if __name__ == '__main__':
    args = get_args_and_initialize()
    group_images_into_frames(args)
    make_video(args.dirpath+'images_for_video/grouped/', '.png')
