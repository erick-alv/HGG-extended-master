import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import glob2
import argparse
import pandas as pd

def load_results(file):
    if not os.path.exists(file):
        return None
    pd_data = pd.read_csv(file)
    N_success_rate = pd_data[['N', 'Success']]
    success_rate_grouped = N_success_rate.groupby(['N'])
    N_configs = success_rate_grouped.indices.keys()
    success_rate_mean = success_rate_grouped.mean()['Success']#this calculates the mean
    success_rate_mean = success_rate_mean.base[0]
    if len(success_rate_grouped.indices[0]) == 1:#this means it was measured with single run
        success_rate_std = np.zeros(shape=(len(success_rate_mean)))
    else:
        success_rate_std = success_rate_grouped.std()['Success']
        success_rate_std = success_rate_std.base[:, 0]
        success_rate_grouped.groups[0]

    return N_configs, success_rate_mean, success_rate_std

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

if __name__ == "__main__":
    # call plot.py to plot success stored in progress.csv files

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('env_id', type=str)
    parser.add_argument('--smooth', type=int, default=1)
    parser.add_argument('--naming', type=int, default=0)
    parser.add_argument('--e_per_c', type=int, default=50)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    env_id = args.env_id

    # Load all data.
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'progress.csv'))]
    paths = [p for p in paths if 'TEST-' in p]
    data = {}
    data_std = {}
    configs = []
    groups_keys = None
    groups = [0, 2, 4]

    for i_path, curr_path in enumerate(paths):
        if not os.path.isdir(curr_path):
            continue
        if not args.dir in curr_path:
            continue
        clean_path = curr_path.replace(env_id, '')
        clean_path = os.path.basename(os.path.normpath(clean_path))
        name = curr_path.split('/')[-2]
        if name.startswith('-'):
            name = '\u03B7 =' + name
        config = name

        # Test:
        print('Config : {}'.format(config))
        results = load_results(os.path.join(curr_path, 'progress.csv'))
        if not results:
            print('skipping {}'.format(curr_path))
            continue

        N_groups, success_rate, success_std = results
        if groups_keys is None:
            groups_keys = N_groups
            for g in groups:
                assert g in N_groups
        else:
            assert groups_keys == N_groups


        if config not in configs:
            configs.append(config)
            data[config]=[]
            data_std[config]=[]

        # instead just using N=0,2,4 since others do not change that much
        success_rate = success_rate[:-2]
        success_std = success_std[:-2]


        data[config].append(success_rate)
        data_std[config].append(success_std)
    rects_data = []
    configs.sort()
    for config in configs:
        #calculation cumulate mean
        mean = np.mean(data[config], axis=0)
        #calculation cumulated STD
        diffs_sq = [np.power(a - mean, 2) for a in data[config]]
        prev_sum_var = np.mean([np.power(s, 2) for s in data_std[config]], axis=0)
        var = np.mean(diffs_sq, axis=0) + prev_sum_var
        std = np.sqrt(var)
        rects_data.append({'mean':mean,
                           'std':std})

    width = 0.5 / len(groups)
    fig, ax = plt.subplots()
    ind = np.arange(len(groups))

    #here make per config
    for i, config in enumerate(configs):
        r = ax.bar(ind + i*width, rects_data[i]['mean'], width, label=config, yerr=rects_data[i]['std'])

    ax.set_ylabel('Success rate best policy',fontsize=18)
    ax.set_title('Rates with tolerance of N collisions', fontsize=18)


    ax.legend(loc=4,prop={'size': 14})
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=16)

    ax.set_xticks(ind)
    ticks_labels = []
    for g in groups:
        ticks_labels.append('N={}'.format(g))
    ax.set_xticklabels(ticks_labels)

    plt.tight_layout()
    plt.savefig(os.path.join(args.dir, 'fig_bar{}.pdf'.format(env_id)), format='pdf')
    if args.save_path:
        plt.savefig(args.save_path)