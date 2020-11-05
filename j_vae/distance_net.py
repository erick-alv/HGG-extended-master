from envs.distance_graph import DistanceGraph2D
import numpy as np

def gen_training_data(N):
    def get_data_points_pairs():
        pass

    #up to 10 bboxes;
    def generate_bounding_boxes(field):
        max_boxes = 10
        n_pos = np.linspace(start=0, stop=10, endpoint=True, num=11)
        num_boxes = np.random.choice(n_pos, p=[0.01,]+10*[0.099])
        g_bboxes = np.zeros(shape=(max_boxes, 4))
        for i in range(num_boxes):
            b = gen_single_box_inside_field(field)
            g_bboxes[i] = b

        #changes order of boxes
        np.random.shuffle(g_bboxes)
        pass

    def gen_single_box(possible_begin_x, possible_begin_y, possible_lengths_x, possible_lengths_y):
        field_begin_x = np.random.uniform(low=possible_begin_x[0], high=possible_begin_x[1])
        field_begin_y = np.random.uniform(low=possible_begin_y[0], high=possible_begin_y[1])
        lx = np.random.choice(possible_lengths_x)
        center_x = field_begin_x + lx / 2.
        ly = np.random.choice(possible_lengths_y)
        center_y = field_begin_y + ly / 2.
        box = [center_x, center_y, lx / 2., ly / 2.]
        return box

    def gen_single_box_inside_field(field):
        #here lx and ly are half the size
        center_x, center_y, lx, ly = field
        begin_x = np.random.uniform(low=center_x-lx, high=center_x+lx)
        begin_y = np.random.uniform(low=center_y-ly, high=center_y+ly)
        #possible lengths. Do not allow too small since graph cannot handle that case
        if lx > 1:
            nx=20
        else:
            nx = 10
        possible_x = np.linspace(start=lx/nx, stop=2*lx, endpoint=True, num=nx)
        #this is the whole lenght
        blx = np.random.choice(possible_x)
        cx = begin_x + blx /2.

        if ly > 1:
            ny=20
        else:
            ny = 10
        possible_y = np.linspace(start=ly/ny, stop=2*ly, endpoint=True, num=ny)
        bly = np.random.choice(possible_y)
        cy = begin_y + bly / 2.

        return np.array([cx, cy, blx/2., bly/2.])



    # generate region
    pos_fields_begin = [-2,2]
    pos_f_l = np.linspace(start=0.5, stop=3, endpoint=True, num=6)
    field = gen_single_box(possible_begin_x=pos_fields_begin, possible_begin_y=pos_fields_begin,
                           possible_lengths_x=pos_f_l, possible_lengths_y=pos_f_l)
    generated_bounding_boxes = generate_bounding_boxes(field)



    #generate points to measure distances (can surpass, (-1, 1) so it learns such cases are far) actually use region also to esitmate distance

    #generate bounding boxes

    #generate graph and process graph

    #calculate distances between the points and save it a file

    '''graph = DistanceGraph2D(args=None, field=field, num_vertices=num_vertices,
                            obstacles=obstacles, size_increase=size_increase)
    graph.compute_cs_graph()
    graph.compute_dist_matrix()
    self.graph = graph'''

    pass

if __name__ == '__main__':
    gen_training_data(30)