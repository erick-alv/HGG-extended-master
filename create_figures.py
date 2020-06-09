import sys
import numpy as np
from envs.distance_graph import DistanceGraph
import glob
import pickle
import matplotlib.pyplot as plt
from utils.os_utils import get_arg_parser

def figure_1(figure="a", save_dir="figures"):
    print("Creating figure 1{} ...".format(figure))
    save_path = save_dir + "/Figure_1" + figure
    adapt_dict = dict()
    # FetchPushObstacle modified!
    adapt_dict["field"] = [1.3, 0.75, 0.6, 0.25, 0.35, 0.2]
    adapt_dict["obstacles"] = [
        [1.3, 0.75, 0.6 - 0.1, 0.25, 0.05, 0.1]]
    num_vertices = [21, 21, 11]  # [20, 20, 10]
    goal_1a = [1.5, 0.645, 0.44]
    goal_1b = [1.5, 0.75, 0.68]
    goal_2 = [1.5, 0.855, 0.44]
    graph = DistanceGraph(None, field=adapt_dict["field"], num_vertices=num_vertices, obstacles=adapt_dict["obstacles"])
    graph.compute_cs_graph()
    graph.compute_dist_matrix(compute_predecessors=True)
    if figure == "a":
        dist, path = graph.get_dist(goal_1a, goal_2, return_path=True)
        goals = [goal_1a, goal_2]
        print("Dist: {}".format(dist))
        graph.plot_graph(goals=goals, obstacle_vertices=False, graph=False, path=path, save_path=save_path)
    elif figure ==  "b":
        dist, path = graph.get_dist(goal_1b, goal_2, return_path=True)
        goals = [goal_1b, goal_2]
        print("Dist: {}".format(dist))
        graph.plot_graph(goals=goals, obstacle_vertices=False, graph=False, path=path, save_path=save_path)
    elif figure == "c":
        goals = [goal_1a, goal_2]
        graph.plot_graph(goals=goals, obstacle_vertices=False, graph=False, path=goals, save_path=save_path)
    elif figure == "d":
        goals = [goal_1b, goal_2]
        graph.plot_graph(goals=goals, obstacle_vertices=False, graph=False, path=goals, save_path=save_path)
    else:
        raise Exception("Only Figures 1a-d!")
    print("... saved to: {}.pdf".format(save_path))

def figure_2(figure="a", save_dir="figures"):
    print("Creating figure 2{} ...".format(figure))
    save_path = save_dir + "/Figure_2" + figure
    adapt_dict = dict()
    # FetchPushObstacle modified!
    adapt_dict["field"] = [1.3, 0.75, 0.6, 0.25, 0.35, 0.2]
    adapt_dict["obstacles"] = [[1.3, 0.75, 0.6 - 0.1, 0.25, 0.2, 0.1]]
    num_vertices = [4, 4, 4]
    graph = DistanceGraph(None, field=adapt_dict["field"], num_vertices=num_vertices, obstacles=adapt_dict["obstacles"])
    graph.compute_cs_graph()
    graph.compute_dist_matrix(compute_predecessors=True)
    if figure == "a":
        print("HI")
        graph.plot_graph(obstacle_vertices=True, graph=False, azim=10, elev=16, save_path=save_path)
    elif figure == "b":
        graph.plot_graph(obstacle_vertices=True, graph=True, azim=10, elev=16, save_path=save_path)
    elif figure == "c":
        goal_1 = [1.5, 0.45, 0.405]
        goal_2 = [1.4, 1.05, 0.405]
        goals = [goal_1, goal_2]
        dist, path = graph.get_dist(goal_1, goal_2, return_path=True)
        print("Dist: {}".format(dist))
        print("Path: {}".format(path))
        graph.plot_graph(obstacle_vertices=True, graph=True, goals=goals, path=path, azim=10, elev=16, save_path=save_path)
    elif figure == "d":
        goal_1 = [1.5, 0.65, 0.78]
        goal_2 = [1.4, 1.05, 0.405]
        goals = [goal_1, goal_2]
        dist, path = graph.get_dist(goal_1, goal_2, return_path=True)
        print("Dist: {}".format(dist))
        graph.plot_graph(obstacle_vertices=True, graph=True, goals=goals, path=path, azim=10, elev=16, save_path=save_path)
    elif figure == "e":
        goal_1 = [1.5, 0.75, 0.405]
        goal_2 = [1.4, 1.05, 0.405]
        goals = [goal_1, goal_2]
        dist, path = graph.get_dist(goal_1, goal_2, return_path=True)
        print("Dist: {}".format(dist))
        graph.plot_graph(obstacle_vertices=True, graph=True, goals=goals, path=path, azim=10, elev=16, save_path=save_path)
    elif figure == "f":
        goal_1 = [1.5, 0.35, 0.405]
        goal_2 = [1.4, 1.05, 0.405]
        goals = [goal_1, goal_2]
        dist, path = graph.get_dist(goal_1, goal_2, return_path=True)
        print("Dist: {}".format(dist))
        graph.plot_graph(obstacle_vertices=True, graph=True, goals=goals, path=path, azim=10, elev=16, save_path=save_path)
    else:
        raise Exception("Only Figures 2a-f!")
    print("... saved to: {}.pdf".format(save_path))

def figure_4(figure="a", save_dir="figures"):
    print("Creating figure 4{} ...".format(figure))
    save_path = save_dir + "/Figure_4" + figure
    if figure == "a":
        goal_filepath = "figures/goals_laby/*.pkl"
    elif figure == "b":
        goal_filepath = "figures/goals_laby_hgg/*.pkl"
    else:
        raise Exception("Only Figures 4a-b!")
    goal_list = []
    for filepath in sorted(glob.glob(goal_filepath)):
        print(filepath)
        file = open(filepath, "rb")
        goal = pickle.load(file)
        goal_list.append(goal)
    color_list = ["red", "orange", "yellow", "green"]
    adapt_dict = dict()
    adapt_dict["field"] = [1.3, 0.75, 0.5, 0.25, 0.35, 0.1]
    adapt_dict["obstacles"] = [[1.3 - 0.1, 0.75, 0.6 - 0.1, 0.11, 0.02, 0.1],
                               [1.3 - 0.23, 0.75, 0.6 - 0.1, 0.02, 0.35, 0.1],
                               [1.3 + 0.03, 0.75, 0.6 - 0.1, 0.02, 0.2, 0.1]]
    num_vertices = [31, 31, 11]
    graph = DistanceGraph(None, field=adapt_dict["field"], num_vertices=num_vertices, obstacles=adapt_dict["obstacles"])
    graph.plot_goals(goals=goal_list, colors=color_list, azim=-56, elev=28, save_path=save_path)
    print("... saved to: {}.pdf".format(save_path))

def figure_3(save_dir="figures"):
    print("Creating figure 3 ...")
    save_path = save_dir + "/Figure_3"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                ax.plot([0, i], [0, j], [0, k], c=[0, 0, 0, 0.3])

    for i in [-2, -1, 0, 1, 2]:
        for j in [-2, -1, 0, 1, 2]:
            for k in [-2, -1, 0, 1, 2]:
                if abs(i) == 0 and abs(j) == 0 and abs(k) == 0:
                    ax.scatter([i], [j], [k], c=[0, 0, 0, 0.8])
                elif abs(i) == 2 or abs(j) == 2 or abs(k) == 2:
                    a = 1
                    # ax.scatter([i], [j], [k], c=[0, 0, 0, 0.2])
                else:
                    if k == -1:
                        ax.scatter([i], [j], [k], c=[0, 0, 1, 0.8])
                    elif k == 0:
                        ax.scatter([i], [j], [k], c=[1, 0, 1, 0.8])
                    else:
                        ax.scatter([i], [j], [k], c=[1, 0, 0, 0.8])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=16, azim=10)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zticks([-1, 0, 1])
    plt.savefig(save_path+".pdf")
    print("... saved to: {}.pdf".format(save_path))

def all():
    figure_1(figure="a")
    figure_1(figure="b")
    figure_1(figure="c")
    figure_1(figure="d")
    figure_2(figure="a")
    figure_2(figure="b")
    figure_2(figure="c")
    figure_2(figure="d")
    figure_2(figure="e")
    figure_2(figure="f")
    figure_3()
    figure_4(figure="a")
    figure_4(figure="b")

if __name__ == "__main__":
    all()