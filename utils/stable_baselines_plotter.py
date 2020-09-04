# from  https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/results_plotter.py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
plt.rcParams['svg.fonttype'] = 'none'

X_TIMESTEPS = 'timesteps'
X_EPISODES = 'episodes'
X_WALLTIME = 'walltime_hrs'
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
          'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
          'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']


def rolling_window(array, window):
    """
    apply a rolling window to a np.ndarray
    :param array: (np.ndarray) the input Array
    :param window: (int) length of the rolling window
    :return: (np.ndarray) rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def window_func(var_1, var_2, window, func):
    """
    apply a function to the rolling window of 2 arrays
    :param var_1: (np.ndarray) variable 1
    :param var_2: (np.ndarray) variable 2
    :param window: (int) length of the rolling window
    :param func: (numpy function) function to apply on the rolling window on variable 2 (such as np.mean)
    :return: (np.ndarray, np.ndarray)  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1:], function_on_var2


    '''def ts2xy(timesteps, xaxis):
    """
    Decompose a timesteps variable to x ans ys
    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: (np.ndarray, np.ndarray) the x and y output
    """
    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
        y_var = timesteps.r.values
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
        y_var = timesteps.r.values
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
        y_var = timesteps.r.values
    else:
        raise NotImplementedError
    return x_var, y_var'''


def plot_curves(xy_list, xlabel, ylabel, window=1, labels=None,title=None, filename=None):
    """
    plot the curves
    :param xy_list: ([(np.ndarray, np.ndarray)]) the x and y coordinates to plot
    :param x_label: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: (str) the title of the plot
    """
    plt.figure(figsize=(16, 8))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        #plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= window:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, window, np.mean)
            if labels is None:
                plt.plot(x, y_mean, color=color)
            else:
                plt.plot(x, y_mean, color=color, label =labels[i])
    plt.xlim(minx, maxx)
    if title is not None:
        plt.title(title)
    plt.legend(loc="upper left")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)


'''def plot_results(dirs, num_timesteps, xaxis, task_name):
    """
    plot the results
    :param dirs: ([str]) the save location of the results to plot
    :param num_timesteps: (int or None) only plot the points below this value
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: (str) the title of the task to plot
    """

    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)
    xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
    plot_curves(xy_list, xaxis, task_name)'''

'''def main():
    """
    Example usage in jupyter-notebook
    .. code-block:: python
        from stable_baselines import results_plotter
        %matplotlib inline
        results_plotter.plot_results(["./log"], 10e6, results_plotter.X_TIMESTEPS, "Breakout")
    Here ./log is a directory containing the monitor.csv files
    """
    import argparse
    import os
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dirs', help='List of log directories', nargs='*', default=['./log'])
    parser.add_argument('--num_timesteps', type=int, default=int(10e6))
    parser.add_argument('--xaxis', help='Varible on X-axis', default=X_TIMESTEPS)
    parser.add_argument('--task_name', help='Title of plot', default='Breakout')
    args = parser.parse_args()
    args.dirs = [os.path.abspath(folder) for folder in args.dirs]
    plot_results(args.dirs, args.num_timesteps, args.xaxis, args.task_name)
    plt.show()'''

def tdm_training_plot():
    df = pd.read_csv('../logsdir/csv_logs/2020_07_02_12_18_26tdm_training_log.csv')
    indexes = df.columns.values
    '''df.dropna(subset=[i for i in indexes if i != 'epoch'],
              inplace=True, how='all')#this is just because logger creates empty row after before each epoch'''

    epoch_time = df[['epoch', 'time']].groupby(by=['epoch'])
    epoch_time = epoch_time.last()
    y = epoch_time[['time']].values[:, 0]
    x = np.arange(0, len(y))
    plot_curves([(x, y)], 'epoch', 'time needed', title='(mixed)',
                filename='../logsdir/csv_logs/time.png')

    epAndEpoch_losses = df[
        ['epoch', 'episode', 'actor_tr_loss', 'critic_tr_loss', 'actor_eval_loss', 'critic_eval_loss']
    ].dropna(subset=['episode']).fillna(0.)
    x = np.arange(0, len(epAndEpoch_losses))
    losses_labels = ['actor_tr_loss', 'critic_tr_loss', 'actor_eval_loss', 'critic_eval_loss']
    xs_ys = [(x, epAndEpoch_losses[[k]].values[:, 0])
             for k in losses_labels]
    plot_curves(xs_ys, 'episode (each 100 timesteps)', 'loss', window=20, title='accumulated loss (mixed)',
                labels=losses_labels, filename='../logsdir/csv_logs/log_losses.png')

    epAndEpoch_reward = df[['epoch', 'episode', 'episode_reward']].dropna(subset=['episode'])
    epAndEpoch_distance = df[['epoch', 'episode', 'distance_to_goal']].dropna(subset=['episode'])

    y = epAndEpoch_distance[['distance_to_goal']].values[:, 0]
    x = np.arange(0, len(y))
    plot_curves([(x, y)], 'episode', 'distance to goal (euclidean distance)', title='distance after episode (mixed)',
                window=20, filename='../logsdir/csv_logs/distance.png')

def plot_distances(df, columns_names, labels, name):
    x = np.arange(0, len(df))
    xs_ys = [(x, df[[k]].values[:, 0]) for k in columns_names]
    plot_curves(xs_ys, 'step', 'distance ', title='distance '+name,
                labels=labels, filename='../logsdir/csv_logs/distance_evaluation_'+name+'.png')



def plot_against(path_to_folder,csv_filename_prefix, route1_name, route2_name, epoch):
    df1 = pd.read_csv(path_to_folder + csv_filename_prefix+route1_name + '.csv')
    df2 = pd.read_csv(path_to_folder +csv_filename_prefix+ route2_name + '.csv')
    def plot_ag(columns_names, labels):
        x = np.arange(0, len(df1))
        xs_ys = [(x, df1[[k]].values[:, 0]) for k in columns_names]
        xs_ys2 = [(x, df2[[k]].values[:, 0]) for k in columns_names]
        l1 = [l + '_'+route1_name for l in labels]
        l2 = [l + '_'+route2_name for l in labels]
        plot_curves(xs_ys + xs_ys2, 'step', 'distance ', title='distance ' +route1_name+'_against_'+route2_name,
                    labels=l1 + l2,
                    filename=path_to_folder + route1_name+'_against_'+
                             route2_name+'_'+'_'.join(labels) +'_epoch{}'.format(epoch)+'.png')
    columns_names_goal = ['from_achieved_goal_to_desired_goal_l2_dist']
    labels_goal = ['l2_distance']
    plot_ag(columns_names_goal, labels_goal)

    columns_names_goal = ['from_state_latent_to_goal_latent_estimator1']
    labels_goal = ['estimator1']
    plot_ag(columns_names_goal, labels_goal)

    columns_names_goal = ['from_state_latent_to_goal_latent_estimator2']
    labels_goal = ['estimator2']
    plot_ag(columns_names_goal, labels_goal)


def plot_distance_estimators(df, name_prefix):
    #to goal
    plot_distances(df, ['from_achieved_goal_to_desired_goal_l2_dist', 'from_state_latent_to_goal_latent_estimator1',
                        'from_state_latent_to_goal_latent_estimator2'],
                   ['l2_distance', 'estimator1', 'estimator2'],
                   name_prefix + '_distance_to_goal')
    plot_distances(df, ['from_achieved_goal_to_desired_goal_l2_dist', 'from_state_latent_to_goal_latent_estimator1'],
                   ['l2_distance', 'estimator1'],
                   name_prefix + '_distance_to_goal_estimator_1')
    plot_distances(df, ['from_achieved_goal_to_desired_goal_l2_dist', 'from_state_latent_to_goal_latent_estimator2'],
                   ['l2_distance', 'estimator2'],
                   name_prefix + '_distance_to_goal_estimator_2')
    # to last state
    plot_distances(df, ['from_achieved_goal_to_achieved_goal_l2_dist', 'from_state_latent_to_state_latent_estimator1',
                        'from_state_latent_to_state_latent_estimator2', 'to_last_steps'],
                   ['l2_distance', 'estimator1', 'estimator2', 'steps_distance'],
                   name_prefix + '_distance_to_last_state')
    plot_distances(df, ['from_achieved_goal_to_achieved_goal_l2_dist', 'from_state_latent_to_state_latent_estimator1',
                        'to_last_steps'],
                   ['l2_distance', 'estimator1', 'steps_distance'],
                   name_prefix + '_distance_to_last_state_estimator_1')
    plot_distances(df, ['from_achieved_goal_to_achieved_goal_l2_dist', 'from_state_latent_to_state_latent_estimator2',
                        'to_last_steps'],
                   ['l2_distance', 'estimator2', 'steps_distance'],
                   name_prefix + '_distance_to_last_state_estimator_2')
    #to last trajectory
    plot_distances(df, ['achieved_goal_l2_dist_traj', 'state_latent_estimator1_traj',
                        'state_latent_estimator2_traj', 'to_last_steps'],
                   ['l2_distance', 'estimator1', 'estimator2', 'steps_distance'],
                   name_prefix + '_along_trajectory_cumulated')
    plot_distances(df, ['achieved_goal_l2_dist_traj', 'state_latent_estimator1_traj',
                        'to_last_steps'],
                   ['l2_distance', 'estimator1', 'steps_distance'],
                   name_prefix + '_along_trajectory_cumulated_estimator_1')
    plot_distances(df, ['achieved_goal_l2_dist_traj', 'state_latent_estimator2_traj',
                        'to_last_steps'],
                   ['l2_distance', 'estimator2', 'steps_distance'],
                   name_prefix + '_along_trajectory_cumulated_estimator_2')
    # to next
    plot_distances(df, ['achieved_goal_to_next_l2_dist', 'state_latent_to_next_estimator1',
                        'state_latent_to_next_estimator2'],
                   ['l2_distance', 'estimator1', 'estimator2'],
                   name_prefix + '_to_next')
    plot_distances(df, ['achieved_goal_to_next_l2_dist', 'state_latent_to_next_estimator2'],
                   ['l2_distance', 'estimator1'],
                   name_prefix + '_to_next_estimator_1')
    plot_distances(df, ['achieved_goal_to_next_l2_dist', 'state_latent_to_next_estimator1'],
                   ['l2_distance', 'estimator2'],
                   name_prefix + '_to_next_estimator_2')

def plot_distance_estimators_from_training(df, name_prefix):
    #to goal
    plot_distances(df, ['to_goal_l2', 'to_goal_est','to_goal_2est'],
                   ['l2_distance', 'estimator1', 'estimator2'],
                   name_prefix + '_distance_to_goal')
    plot_distances(df, ['to_goal_l2', 'to_goal_est'],
                   ['l2_distance', 'estimator1'],
                   name_prefix + '_distance_to_goal_estimator_1')
    plot_distances(df, ['to_goal_l2','to_goal_2est'],
                   ['l2_distance', 'estimator2'],
                   name_prefix + '_distance_to_goal_estimator_2')
    # to last state

    plot_distances(df, ['to_last_l2', 'to_last_est','to_last_2est', 'to_last_steps'],
                   ['l2_distance', 'estimator1', 'estimator2', 'steps_distance'],
                   name_prefix + '_distance_to_last_state')
    plot_distances(df, ['to_last_l2', 'to_last_est', 'to_last_steps'],
                   ['l2_distance', 'estimator1', 'steps_distance'],
                   name_prefix + '_distance_to_last_state_estimator_1')
    plot_distances(df, ['to_last_l2','to_last_2est', 'to_last_steps'],
                   ['l2_distance', 'estimator2', 'steps_distance'],
                   name_prefix + '_distance_to_last_state_estimator_2')
    #to last trajectory
    plot_distances(df, ['to_last_l2_traj', 'to_last_est_traj','to_last_2est_traj', 'to_last_steps'],
                   ['l2_distance', 'estimator1', 'estimator2', 'steps_distance'],
                   name_prefix + '_along_trajectory_cumulated')
    plot_distances(df, ['to_last_l2_traj', 'to_last_est_traj', 'to_last_steps'],
                   ['l2_distance', 'estimator1', 'steps_distance'],
                   name_prefix + '_along_trajectory_cumulated_estimator_1')
    plot_distances(df, ['to_last_l2_traj','to_last_2est_traj', 'to_last_steps'],
                   ['l2_distance', 'estimator2', 'steps_distance'],
                   name_prefix + '_along_trajectory_cumulated_estimator_2')

    # to next
    plot_distances(df, ['to_next_l2','to_next_est','to_next_2est'],
                   ['l2_distance', 'estimator1', 'estimator2'],
                   name_prefix + '_to_next')
    plot_distances(df, ['to_next_l2','to_next_est'],
                   ['l2_distance', 'estimator1'],
                   name_prefix + '_to_next_estimator_1')
    plot_distances(df, ['to_next_l2','to_next_2est'],
                   ['l2_distance', 'estimator2'],
                   name_prefix + '_to_next_estimator_2')

def plot_distance_group(df, name_prefix):
    def p_points(columns_names, labels, name):
        x = np.arange(0, len(df))
        xs_ys = [(x, df[[k]].values[:, 0]) for k in columns_names]
        plot_curves(xs_ys, 'pair', 'distance', title='distance ' + name,
                    labels=labels, filename='../logsdir/csv_logs/distance_evaluation_' + name + '.png')
    p_points(['l2_dist'],['l2_distance'],name_prefix + 'l2_distance')
    p_points(['estimator1'], ['estimator1'],name_prefix + 'estimator1')
    p_points(['estimator2'],['estimator2'],name_prefix + 'estimator2')



if __name__ == "__main__":
    epoch = 390
    '''
    df = pd.read_csv('../logsdir/csv_logs/dist_no_obstacle_epoch_120_it_0.csv')
    plot_distance_estimators_from_training(df,'from_training_epoch_120_it_0')
    '''
    l = ['random','route_1','route_2','route_3_1','route_3_2','route_3_3','route_4_1','route_4_2']
    for r in l:
        df = pd.read_csv('../logsdir/csv_logs/distance_evaluation_'+r+'.csv')
        plot_distance_estimators(df, r+'_epoch_{}_'.format(epoch))

    plot_against('../logsdir/csv_logs/','distance_evaluation_','route_3_1','route_3_2', epoch)
    plot_against('../logsdir/csv_logs/', 'distance_evaluation_', 'route_4_1', 'route_4_2', epoch)

    for gr in ['a','b','c','d','e','f', 'g', 'h', 'i']:
        df = pd.read_csv('../logsdir/csv_logs/group_'+gr+'.csv')
        plot_distance_group(df, 'group_'+gr+'_epoch_{}_'.format(epoch))



