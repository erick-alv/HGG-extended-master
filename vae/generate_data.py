import argparse
import numpy as np
import os
from envs import make_temp_env, Robotics_envs_id
from utils.os_utils import make_dir, str2bool
from vae_env_inter import take_objects_image_training
from PIL import Image
from scipy.spatial.transform import Rotation
from utils.calculation_utils import interval_map_function
import csv
import io

#make range bigger than table so there are samples falling and in other positions
extend_region_parameters_goal = {'FetchGenerativeEnv-v1': {'center':np.array([1.3, 0.75, 0.425]), 'range':0.25},
                                 }


def extend_sample_region(env, args):
    env.object_center = extend_region_parameters_goal[args.env]['center']
    #todo?? is this necessary
    env.obj_range = extend_region_parameters_goal[args.env]['range']


def _get_reduce_factor(distance_start_reduce, max_size, max_size_at_edge, current_size):
    if current_size <= max_size_at_edge:
        return 0.
    else:
        reduce_prc = (current_size - max_size_at_edge) / (max_size - max_size_at_edge)
        return distance_start_reduce * reduce_prc


def _get_pos(ar_x, ar_y, center, range_left, range_right, range_up, range_down, size):
    def _gen_pos():
        p = center
        p[0] += np.random.uniform(range_left, range_right, size=1)
        p[1] += np.random.uniform(range_up, range_down, size=1)
        p[2] = 0.4 + size[2]
        return p

    def is_inside_ocuped_areas(p, s):
        for i in range(len(ar_x)):
            if (ar_x[i][0] <= p[0] + s[0] <= ar_x[i][1] and ar_y[i][0] <= p[1] + s[1] <= ar_y[i][1]):
                return True
            elif (ar_x[i][0] <= p[0] - s[0] <= ar_x[i][1] and ar_y[i][0] <= p[1] - s[1] <= ar_y[i][1]):
                return True
            elif (ar_x[i][0] <= p[0] <= ar_x[i][1] and ar_y[i][0] <= p[1] <= ar_y[i][1]):
                return True
        return False

    t = 0
    while True:
        t += 1
        pos = _gen_pos()
        '''try:
            assert pos[0] >= 1.025
            assert pos[0] <= 1.575
            assert pos[1] >= 0.475
            assert pos[1] <= 1.025
        except:
            print('pos:, {} \ncenter:, {} \nrange_left:, {} \nrange right:, {} \nrange up:, {} \nrange down:, {} \nsize:, {} \n'.
                format(pos, center, range_left, range_right, range_up, range_down, size))
            exit()'''
        if not is_inside_ocuped_areas(pos, [size[0]+0.01, size[1]+0.01]):
            return pos
        if t >= 100:
            print('took to much time')
            return None


def _gen_cylinder(ocuped_areas_x, ocuped_areas_y):
    #size
    max_r = 0.08
    max_at_edge = 0.05
    min_size = 0.035
    start_reduce = 0.03

    r = np.random.uniform(min_size, max_r)
    height = np.random.uniform(0.02, 0.035)
    size = [r, height, 0.0]


    # rotation
    rot_x = np.random.uniform(0., 180.)
    rot_y = np.random.uniform(0., 180.)


    #position
    reduce_factor = _get_reduce_factor(start_reduce, max_r, max_at_edge, r)
    assert reduce_factor >= 0.
    assert env.obj_range - reduce_factor > 0
    pos = _get_pos(ocuped_areas_x, ocuped_areas_y, env.object_center.copy(), -env.obj_range + reduce_factor,
                   env.obj_range - reduce_factor, -env.obj_range + reduce_factor, env.obj_range - reduce_factor,
                   [r, r, height])#for the cylinder emulates a square bounding box
    if pos is None:
        return None, None
    else:
        # generate bounding box
        # First as if object would be at origin
        s_x_v = np.zeros(3)
        s_x_v[0] = size[0]
        s_y_v = np.zeros(3)
        s_y_v[1] = size[0]
        s_z_v = np.zeros(3)
        s_z_v[2] = size[1]
        vertices = []
        for xmz in [-1, 1]:
            for ymz in [-1, 1]:
                for zmz in [-1, 1]:
                    vertices.append(0. + xmz * s_x_v + ymz * s_y_v + zmz * s_z_v)
        # rotation matrix
        # todo verify if here is also necessary to cahnge rot_z and rot_x
        r = Rotation.from_rotvec(np.pi / 180 * np.array([0., rot_y, rot_x]))
        for i, v in enumerate(vertices):
            #todo correct rotation
            #tr_p = r.apply(v + pos)
            #vertices[i] = tr_p
            vertices[i] = r.apply(v) + pos
        vertices = np.array(vertices)
        bbox_min_x = np.min(vertices[:, 0])
        bbox_max_x = np.max(vertices[:, 0])
        dist_x = np.abs(bbox_max_x - bbox_min_x)
        bbox_min_y = np.min(vertices[:, 1])
        bbox_max_y = np.max(vertices[:, 1])
        dist_y = np.abs(bbox_max_y - bbox_min_y)
        bbox = [(bbox_max_x + bbox_min_x) / 2., (bbox_max_y + bbox_min_y) / 2., dist_x / 2., dist_y / 2.]
        return size, rot_x, rot_y, pos, [pos[0] - size[0], pos[0] + size[0]], [pos[1] - size[1], pos[1] + size[1]], bbox


def _gen_cube(ocuped_areas_x, ocuped_areas_y):
    #size
    max_l = 0.05
    max_at_edge = 0.035
    min_size = 0.03
    start_reduce = 0.015
    s = np.random.uniform(min_size, max_l)
    size = [s, s, s]


    # rotation
    rot_x = np.random.uniform(0., 90.)
    rot_y = np.random.uniform(0., 90.)
    rot_z = np.random.uniform(0., 90.)

    #position
    reduce_factor = _get_reduce_factor(start_reduce, max_l, max_at_edge, s)
    assert reduce_factor >= 0.
    assert env.obj_range - reduce_factor > 0
    pos = _get_pos(ocuped_areas_x, ocuped_areas_y, env.object_center.copy(), -env.obj_range + reduce_factor,
                   env.obj_range - reduce_factor, -env.obj_range + reduce_factor, env.obj_range - reduce_factor,
                   size)
    if pos is None:
        return None, None
    else:
        # generate bounding box
        # First as if object would be at origin
        s_x_v = np.zeros(3)
        s_x_v[0] = size[0]
        s_y_v = np.zeros(3)
        s_y_v[1] = size[1]
        s_z_v = np.zeros(3)
        s_z_v[2] = size[2]
        vertices = []
        for xmz in [-1, 1]:
            for ymz in [-1, 1]:
                for zmz in [-1, 1]:
                    vertices.append(0.+xmz*s_x_v+ymz*s_y_v+zmz*s_z_v)
        #rotation matrix
        #todo verify if here is also necessary to cahnge rot_z and rot_x
        r = Rotation.from_rotvec(np.pi / 180 * np.array([rot_z, rot_y, rot_x]))
        for i, v in enumerate(vertices):
            #todo correct rotation
            #tr_p = r.apply(v + pos)
            #vertices[i] = tr_p
            vertices[i] = r.apply(v) + pos
        vertices = np.array(vertices)
        bbox_min_x = np.min(vertices[:, 0])
        bbox_max_x = np.max(vertices[:, 0])
        dist_x = np.abs(bbox_max_x - bbox_min_x)
        bbox_min_y = np.min(vertices[:, 1])
        bbox_max_y = np.max(vertices[:, 1])
        dist_y = np.abs(bbox_max_y - bbox_min_y)
        bbox = [(bbox_max_x + bbox_min_x)/2., (bbox_max_y + bbox_min_y)/2., dist_x/2., dist_y/2.]


        return size, rot_x, rot_y, rot_z, pos, [pos[0] - size[0], pos[0] + size[0]], \
               [pos[1] - size[1], pos[1] + size[1]], bbox


def _gen_rectangle():
    # size obstacle
    max_l = 0.25
    min_l = 0.08
    long_length = np.random.uniform(min_l, max_l)
    short_length = np.random.uniform(0.02, 0.04)
    height_obstacle = np.random.uniform(0.02, 0.035)
    size = [long_length, short_length, height_obstacle]

    # rotation obstacle
    r_case = np.random.randint(0, 2)
    if r_case == 0:
        rot_z = 0.
    else:
        rot_z = 90.

    # pos_obstacle
    begin_reduce = 0.2  # distance to edge when when lenght 0.25
    reduce_prc = long_length / max_l
    reduce_factor = begin_reduce * reduce_prc
    pos_obstacle = env.object_center.copy()
    if rot_z == 0.:
        pos_obstacle[0] += np.random.uniform(-env.obj_range + reduce_factor, env.obj_range - reduce_factor, size=1)
        pos_obstacle[1] += np.random.uniform(-env.obj_range, env.obj_range, size=1)
        occuped_area_x = [pos_obstacle[0] - long_length, pos_obstacle[0] + long_length]
        occuped_area_y = [pos_obstacle[1] - short_length, pos_obstacle[1] + short_length]
        bbox = [pos_obstacle[0], pos_obstacle[1], long_length, short_length]
    else:
        pos_obstacle[0] += np.random.uniform(-env.obj_range, env.obj_range, size=1)
        pos_obstacle[1] += np.random.uniform(-env.obj_range + reduce_factor, env.obj_range - reduce_factor, size=1)
        occuped_area_x = [pos_obstacle[0] - short_length, pos_obstacle[0] + short_length]
        occuped_area_y = [pos_obstacle[1] - long_length, pos_obstacle[1] + long_length]
        bbox = [pos_obstacle[0], pos_obstacle[1], short_length, long_length]
    pos_obstacle[2] = 0.4 + height_obstacle
    return size, rot_z, pos_obstacle, occuped_area_x, occuped_area_y, bbox


def bbox_to_image_coordinates(bbox, args):
    x_min = bbox[0] - bbox[2]
    x_max = bbox[0] + bbox[2]
    y_min = bbox[1] - bbox[3]
    y_max = bbox[1] + bbox[3]


    # the distances to the edges seen in the image are 0.025
    map_coords_x = interval_map_function(1.025, 1.575, 0., args.img_size)
    map_coords_y = interval_map_function(0.475, 1.025,  0., args.img_size)
    x_min = map_coords_x(x_min)
    x_min = np.clip(x_min, a_min=0., a_max=args.img_size)
    x_max = map_coords_x(x_max)
    x_max = np.clip(x_max, a_min=0., a_max=args.img_size)
    y_min = map_coords_y(y_min)
    y_min = np.clip(y_min, a_min=0., a_max=args.img_size)
    y_max = map_coords_y(y_max)
    y_max = np.clip(y_max, a_min=0., a_max=args.img_size)

    new_bbox = [x_min, y_min, x_max, y_max]
    if (new_bbox[0] == 0. and new_bbox[2] == 0.) or (new_bbox[1] == 0. and new_bbox[3] == 0.) \
            or (new_bbox[0] == args.img_size and new_bbox[2] == args.img_size) \
            or (new_bbox[1] == args.img_size and new_bbox[3] == args.img_size):
        return None
    try:
        assert y_max != y_min
        assert y_max > y_min
        assert x_max != x_min
        assert x_max > x_min
        assert new_bbox != [0., 0., 0., 0.]
    except:
        pass

    # here the y coordinates are flipped since images use other direction
    new_y_max = args.img_size - y_min
    new_y_min = args.img_size - y_max
    new_bbox = [x_min, new_y_min, x_max, new_y_max]
    return new_bbox


def gen_all_data_mixed(env, args):
    for n in ['rectangle', 'rectangle1', 'rectangle2', 'cylinder', 'cylinder1', 'cube', 'cube1']:
        env.env.env._set_position(names_list=[n], position=[10., 10., 0.])

    colors = [(1., 0., 0.), (0., 0., 1.), (0.5, 0., 0.5), (0.8, 0.8, 0.)]
    color_count = {}
    for i in range(len(colors)):
        color_count[i] = 0

    def select_and_change_color(obj_name):
        color_ind = np.random.randint(0, len(colors))
        while color_count[color_ind] >= 1:
            color_ind = (color_ind +1) % len(colors)
        r, g, b = colors[color_ind]
        env.env.env._change_color([obj_name], r, g, b)
        color_count[color_ind] +=1

    int_codes_objects = {'rectangle': 1, 'cylinder': 2, 'cube': 3}
    bboxes = []
    els_labels = []
    occuped_areas_x = []
    occuped_areas_y = []
    max_els = 4

    if args.save_bbox_data:
        #bboxes for faster rcnn cannot handle empty images
        number_rectangles = np.random.choice(a=[1, 2, 3], p=[0.4, 0.3, 0.3])
    else:
        number_rectangles = np.random.choice(a=[0, 1, 2, 3], p=[0.3, 0.2, 0.3, 0.2])
    rem_els = max_els - number_rectangles
    for i in range(number_rectangles):
        if i == 0:
            rect_name = 'rectangle'
        elif i == 1:
            rect_name = 'rectangle1'
        else:
            rect_name = 'rectangle2'

        while True:
            size, rot_z, pos_obstacle, occuped_area_x_rectangle, occuped_area_y_rectangle, bbox = _gen_rectangle()
            bbox = bbox_to_image_coordinates(bbox, args)
            if bbox is not None:
                break
        env.env.env._set_size(names_list=[rect_name], size=size)
        env.env.env._rotate([rect_name], 0., 0., rot_z)
        env.env.env._set_position(names_list=[rect_name], position=pos_obstacle)
        select_and_change_color(rect_name)
        occuped_areas_x.append(occuped_area_x_rectangle)
        occuped_areas_y.append(occuped_area_y_rectangle)
        bboxes.append(bbox)
        els_labels.append(int_codes_objects['rectangle'])

    # generate other objects
    if rem_els > 0:
        max_n_cylynders = min(2, rem_els)
        n_cylinders = np.random.randint(0, max_n_cylynders+1)
        rem_els = rem_els - n_cylinders
        for i in range(n_cylinders):
            if i == 0:
                cyl_name = 'cylinder'
            elif i == 1:
                cyl_name = 'cylinder1'

            while True:
                size_cyl, rot_x_cyl, rot_y_cyl, pos_cyl, oc_x_cyl, oc_y_cyl, bbox = _gen_cylinder(occuped_areas_x, occuped_areas_y)
                bbox = bbox_to_image_coordinates(bbox, args)
                if bbox is not None:
                    break
            env.env.env._set_size(names_list=[cyl_name], size=size_cyl)
            env.env.env._rotate([cyl_name], rot_x_cyl, rot_y_cyl, 0.)
            env.env.env._set_position(names_list=[cyl_name], position=pos_cyl)
            select_and_change_color(cyl_name)
            occuped_areas_x.append(oc_x_cyl)
            occuped_areas_y.append(oc_y_cyl)
            bboxes.append(bbox)
            els_labels.append(int_codes_objects['cylinder'])

    if rem_els > 0:
        max_n_cubes = min(1, rem_els)
        n_cubes = np.random.randint(0, max_n_cubes + 1)

        for i in range(n_cubes):
            if i == 0:
                cube_name = 'cube'
            elif i == 1:
                cube_name = 'cube1'

            while True:
                size_cube, rot_x_cube, rot_y_cube, rot_z_cube, pos_cube, oc_x_cube, oc_y_cube, bbox = _gen_cube(occuped_areas_x, occuped_areas_y)
                bbox = bbox_to_image_coordinates(bbox, args)
                if bbox is not None:
                    break

            env.env.env._set_size(names_list=[cube_name], size=size_cube)
            env.env.env._rotate([cube_name], rot_x_cube, rot_y_cube, rot_z_cube)
            env.env.env._set_position(names_list=[cube_name], position=pos_cube)
            select_and_change_color(cube_name)

            occuped_areas_x.append(oc_x_cube)
            occuped_areas_y.append(oc_y_cube)

            bboxes.append(bbox)
            els_labels.append(int_codes_objects['cube'])
    if args.save_bbox_data:
        #bboxes for faster rcnn cannot handle empty images
        assert len(bboxes) > 0
    assert len(bboxes) == len(els_labels )
    return bboxes, els_labels


gen_setup_env_ops = {'FetchGenerativeEnv-v1':[extend_sample_region]}

after_env_reset_ops = {'FetchGenerativeEnv-v1':[]}

during_loop_ops = {'FetchGenerativeEnv-v1':[gen_all_data_mixed]}

if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', help='the task for the generation of data', type=str,
                        default='generate', choices=['generate', 'mix', 'show'], required=True)
    #CURRENTLY using just FetchGenerativeEnv-v1; argument is left since it is used in different parts of code
    parser.add_argument('--env', help='gym env id', type=str, default='FetchGenerativeEnv-v1', choices=Robotics_envs_id)
    args, _ = parser.parse_known_args()
    if args.task == 'mix':
        parser.add_argument('--file_1', help='first file to mix', type=str)
        parser.add_argument('--file_2', help='second file to mix', type=str)
        parser.add_argument('--output_file', help='name of output file for mixed dataset', type=str)
        args = parser.parse_args()

        this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        base_data_dir = this_file_dir + '../data/'
        env_data_dir = base_data_dir + args.env + '/'
        make_dir(env_data_dir, clear=False)
        data_file_1 = env_data_dir + args.file_1
        data_file_2 = env_data_dir + args.file_2

        data_1 = np.load(data_file_1)
        data_2 = np.load(data_file_2)
        mixed_data = np.concatenate([data_1, data_2], axis=0)
        np.random.shuffle(mixed_data)

        output_file = env_data_dir + args.output_file
        np.save(output_file, mixed_data)

    else:
        if args.env == 'HandReach-v0':
            parser.add_argument('--goal', help='method of goal generation', type=str, default='reach',
                                choices=['vanilla', 'reach'])
        args, _ = parser.parse_known_args()
        if args.env == 'HandReach-v0':
            parser.add_argument('--goal', help='method of goal generation', type=str, default='reach',
                                choices=['vanilla', 'reach'])
        else:
            parser.add_argument('--goal', help='method of goal generation', type=str, default='interval',
                                choices=['vanilla', 'fixobj', 'interval', 'custom'])
            if args.env[:5] == 'Fetch':
                parser.add_argument('--init_offset', help='initial offset in fetch environments', type=np.float32,
                                    default=1.0)
            elif args.env[:4] == 'Hand':
                parser.add_argument('--init_rotation', help='initial rotation in hand environments', type=np.float32,
                                    default=0.25)

        parser.add_argument('--count', help='number of samples', type=np.int32, default=1280*30)
        parser.add_argument('--img_size', help='size image in pixels', type=np.int32, default=64)

        parser.add_argument('--save_im_folder', help='create images subfolder to store images', type=str2bool, default=False)
        parser.add_argument('--save_bbox_data', help='Creates a csv file with bounding Box information; just for FCNN', type=str2bool, default=False)
        args = parser.parse_args()

        # create data folder if it does not exist, corresponding folders, and files where to store data
        this_file_dir = os.path.dirname(os.path.abspath(__file__)) + '/'
        base_data_dir = this_file_dir + '../data/'
        env_data_dir = base_data_dir + args.env + '/'
        make_dir(env_data_dir, clear=False)
        data_file = env_data_dir + 'all_set.npy'

        if args.task == 'generate':
            #load environment
            env = make_temp_env(args)
            #setup env(change generation region; move other objects(just leave those we need)??)
            for func in gen_setup_env_ops[args.env]:
                func(env, args)

            if args.save_bbox_data:
                field_names = ['im_name', 'bbox', 'labels']
                csv_file_path = env_data_dir + 'all_set.csv'
                with open(csv_file_path, 'w') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=field_names)
                    writer.writeheader()
            if args.save_im_folder:
                #makes folder for images
                images_dir = env_data_dir + 'images/'
                make_dir(images_dir, clear=True)


            #loop through(moving object and making)
            train_data = np.empty([args.count, args.img_size, args.img_size, 3])
            i = 0
            while i < args.count:
                env.reset()
                for func in after_env_reset_ops[args.env]:
                    func(env, args)

                for func in during_loop_ops[args.env]:
                    bboxes, els_labels = func(env, args)
                rgb_array = take_objects_image_training(env, img_size=args.img_size)
                train_data[i] = rgb_array.copy()
                if args.save_im_folder:
                    im = Image.fromarray(rgb_array.copy().astype(np.uint8))
                    im.save('{}{}.png'.format(images_dir, i))
                if args.save_bbox_data:
                    buffer_bboxes = io.BytesIO()
                    np.savetxt(buffer_bboxes, np.array(bboxes))
                    buffer_labels = io.BytesIO()
                    np.savetxt(buffer_labels, np.array(els_labels))
                    values = {'im_name':'{}.png'.format(i),
                              'bbox':buffer_bboxes.getvalue().decode('utf-8'),
                              'labels':buffer_labels.getvalue().decode('utf-8')}
                    with open(csv_file_path, 'a') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=field_names)
                        writer.writerow(values)

                i += 1
            np.save(data_file, train_data)
        elif args.task == 'show':
            train_data = np.load(data_file)
            all_idx = np.arange(len(train_data)).tolist()
            def show_some_sampled_images():
                n = 10
                a = None
                spacer_x = np.zeros(shape=(args.img_size, 5, 3))
                spacer_y = np.zeros(shape=(5, n*args.img_size + 5*(n-1) ,3))
                for i in range(n):
                    b = None
                    for j in range(n):
                        id = np.random.choice(all_idx, size=1, replace=False)[0]
                        all_idx.remove(id)
                        j_im = train_data[id].copy()
                        if b is None:
                            b = j_im.copy()
                        else:
                            b = np.concatenate([b, spacer_x, j_im], axis=1)
                    if a is None:
                        a = b.copy()
                    else:
                        a = np.concatenate([a, spacer_y, b], axis=0)
                img = Image.fromarray(a.astype(np.uint8))
                img.save('sample_image_generated.png')
                img.show()
                img.close()

            for _ in range(3):
                show_some_sampled_images()
        else:
            print("No task selected")
