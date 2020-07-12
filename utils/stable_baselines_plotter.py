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


if __name__ == "__main__":
    df = pd.read_csv('../logsdir/csv_logs/2020_07_02_12_18_26tdm_training_log.csv')
    indexes = df.columns.values
    '''df.dropna(subset=[i for i in indexes if i != 'epoch'],
              inplace=True, how='all')#this is just because logger creates empty row after before each epoch'''

    epoch_time = df[['epoch','time']].groupby(by=['epoch'])
    epoch_time = epoch_time.last()
    y = epoch_time[['time']].values[:, 0]
    x = np.arange(0, len(y))
    plot_curves([(x, y)], 'epoch', 'time needed',title='(mixed)',
                filename='../logsdir/csv_logs/time.png')

    epAndEpoch_losses = df[
        ['epoch','episode','actor_tr_loss', 'critic_tr_loss', 'actor_eval_loss','critic_eval_loss']
    ].dropna(subset=['episode']).fillna(0.)
    x = np.arange(0, len(epAndEpoch_losses))
    losses_labels = ['actor_tr_loss', 'critic_tr_loss', 'actor_eval_loss','critic_eval_loss']
    xs_ys = [(x, epAndEpoch_losses[[k]].values[:, 0])
             for k in losses_labels]
    plot_curves(xs_ys, 'episode (each 100 timesteps)', 'loss', window=20, title='accumulated loss (mixed)',
                labels=losses_labels, filename='../logsdir/csv_logs/log_losses.png')

    epAndEpoch_reward = df[['epoch','episode','episode_reward']].dropna(subset=['episode'])
    epAndEpoch_distance = df[['epoch', 'episode', 'distance_to_goal']].dropna(subset=['episode'])

    y = epAndEpoch_distance[['distance_to_goal']].values[:,0]
    x = np.arange(0, len(y))
    plot_curves([(x,y)],'episode', 'distance to goal (euclidean distance)', title='distance after episode (mixed)',
                window=20, filename='../logsdir/csv_logs/distance.png')
