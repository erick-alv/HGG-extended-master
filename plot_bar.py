import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import glob2
import argparse
import pandas as pd

def plot_already_noted():

    N = 4
    #hgg_rates = (0., 0.01, 0.2, 0.78)
    #bbox_rates = (0., 0.08, 0.58, 0.78)
    #optimal_rates = (0., 0.11, 0.71, 0.96)
    hgg_rates = (0., 0.02, 0.26, 0.71)
    bbox_rates = (0., 0.09, 0.49, 0.87)
    bbox_self_v1_rates = (0.02, 0.11, 0.5, 0.7)
    bbox_self_v2_rates = (0.03, 0.21, 0.32, 0.41)
    optimal_rates = (0.02, 0.15, 0.68, 0.96)
    optimal_self_v1_rates = (0.04, 0.14, 0.72, 0.87)
    optimal_self_v2_rates = (0.11, 0.85, 0.95, 0.97)

    #these are values used when trained with stopping then env if there exists some collision
    optimal_self_coll_rates = (0., 0.01, 0.32, 0.94)
    optimal_env_coll_rates = (0.01, 0.13, 0.68, 0.95)
    optimal_self_colls_area_extended1 = (0.02, 0.16, 0.79, 0.95)
    optimal_self_colls_area_extended2 = (0.0, 0.03, 0.65, 0.99)


    optimal_rewMod = (0.23, 0.72, 0.96, 0.98)
    optimal_rewModStop = (0.05, 0.63, 0.98, 0.98)
    optimal_rewModRegion = (0.88, 0.95, 0.97, 0.92)
    optimal_rewModRegionStop = (0.91, 0.96, 0.98, 0.99)
    optimal_regionStop = (0., 0., 0.13, 0.56)

    opt_1180 = (0.15, 0.46, 0.91, 0.98)
    opt_1182 = (0.63, 0.92, 0.98, 0.99)
    opt_1186 = (0.94, 0.96, 0.95, 0.97)
    opt_1280 = (0.1, 0.84, 0.99, 0.99)
    opt_1282 = (0.74, 0.92, 0.96, 0.98)
    opt_1286 = (0.94, 0.97, 0.98, 0.96)

    opt_1176 = (0.97, 0.98, 0.99, 0.99)
    bbox_3180 = (0, 0, 0, 0)
    bbox_3186 = (0.89, 0.96, 0.93, 0.98)

    ind = np.arange(N)
    width = 0.1
    show_results = 4
    if show_results == 0 :
        plt.bar(ind, hgg_rates, width, label='HGG')
        plt.bar(ind + width, bbox_rates, width, label='Bbox')
        plt.bar(ind + 2 * width, bbox_self_v1_rates, width, label='Bbox multi-obj version1')
        plt.bar(ind + 3 * width, bbox_self_v2_rates, width, label='Bbox multi-obj version2')
        plt.bar(ind + 4 * width, optimal_rates, width, label='Optimal')
        plt.bar(ind + 5 * width, optimal_self_v1_rates, width, label='Opt. multi-obj version1')
        plt.bar(ind + 6 * width, optimal_self_v2_rates, width, label='Opt. multi-obj version2')
        figname = 'comparison_all.png'
    elif show_results == 1:
        plt.bar(ind, optimal_rates, width, label='Optimal')
        plt.bar(ind + width, optimal_self_v1_rates, width, label='Opt. multi-obj version1')
        plt.bar(ind + 2 * width, optimal_self_v2_rates, width, label='Opt. multi-obj version2')
        plt.bar(ind + 3 * width, optimal_self_coll_rates, width, label='Opt. coll. self')
        plt.bar(ind + 4 * width, optimal_env_coll_rates, width, label='Opt. coll. env')
        figname = 'comparison_optimal.png'
    elif show_results == 2:
        plt.bar(ind, optimal_rates, width, label='Optimal')
        plt.bar(ind + width, optimal_self_coll_rates, width, label='Opt. coll. with Bbox')
        plt.bar(ind + 2 * width, optimal_env_coll_rates, width, label='Opt. coll. env')
        plt.bar(ind + 3 * width, optimal_self_colls_area_extended1, width,
                label='Opt. coll. with Bbox incremented 0.022')
        plt.bar(ind + 4 * width, optimal_self_colls_area_extended2, width,
                label='Opt. coll. with Bbox incremented 0.045')
        figname = 'comparison_optimal2.png'
    elif show_results == 3:
        plt.bar(ind, optimal_rates, width, label='Optimal')
        plt.bar(ind + width, optimal_rewMod, width, label='mod reward')
        plt.bar(ind + 2 * width, optimal_rewModStop, width, label='mod reward stop')
        plt.bar(ind + 3 * width, optimal_rewModRegion, width,label='mod reward region')
        plt.bar(ind + 4 * width, optimal_rewModRegionStop, width,label='mod reward region stop')
        plt.bar(ind + 5 * width, optimal_regionStop, width, label='region stop')
        figname = 'comparison_optimal_difapr.png'
    elif show_results == 4:
        plt.bar(ind, opt_1180, width, label='1) reward -2')
        plt.bar(ind + width, opt_1182, width, label='1) safe region, reward -2')
        plt.bar(ind + 2 * width, opt_1186, width, label='1) reward -10')
        plt.bar(ind + 3 * width, opt_1280, width,label='2) reward -2')
        plt.bar(ind + 4 * width, opt_1282, width,label='2) safe region, reward -2')
        plt.bar(ind + 5 * width, opt_1286, width, label='2) reward -10')
        figname = 'comparison_rewmods_vs_safe.png'
    elif show_results == 5:
        plt.bar(ind, bbox_3180, width, label='bbox reward -2')
        plt.bar(ind + width, bbox_3186, width, label='bbox reward -10')
        plt.bar(ind + 2 * width, opt_1176, width, label='opt. reward -10 (no stop)')
        plt.bar(ind + 3 * width, opt_1186, width, label='opt. reward -10')
        figname = 'comparison_with_bbox.png'

    plt.ylabel('Success rate best policy')
    plt.title('Rates with tollerance of N collisions')
    plt.xticks(ind + width / 3, ('N=2', 'N=4', 'N=7', 'N=10'))
    plt.legend(loc=4, prop={'size': 8})
    plt.savefig(figname)
    plt.close()

def load_results(file):
    if not os.path.exists(file):
        return None
    pd_data = pd.read_csv(file)
    N_success_rate = pd_data[['N', 'Success']]
    success_rate_grouped = N_success_rate.groupby(['N'])
    N_configs = success_rate_grouped.indices.keys()
    success_rate_mean = success_rate_grouped.mean()['Success']
    success_rate_mean = success_rate_mean.base[0]
    return N_configs, success_rate_mean

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
    data = {}
    paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'progress.csv'))]
    show_results = 4
    rects_data = []
    labels = []
    groups_keys = None

    for i_path, curr_path in enumerate(paths):
        if not os.path.isdir(curr_path):
            continue
        if not args.dir in curr_path:
            continue
        clean_path = curr_path.replace(env_id, '')
        clean_path = os.path.basename(os.path.normpath(clean_path))
        #clean_path = ''.join([i for i in clean_path if not i.isdigit()])
        # divide path into run (number in the beginning) and config (information on configuration, included in the path name)
        if args.naming == 0:
            config = clean_path

        # Test:
        run = config
        print('Config / run: {} / {}'.format(config, run))

        results = load_results(os.path.join(curr_path, 'progress.csv'))
        if not results:
            print('skipping {}'.format(curr_path))
            continue
        N_configs, success_rate_mean = results
        rects_data.append(success_rate_mean)
        labels.append(run)
        if groups_keys is None:
            groups_keys = N_configs
            groups = []
            for k in groups_keys:
                groups.append(k)
        else:
            assert groups_keys == N_configs

    width = 0.6 / len(groups)
    fig, ax = plt.subplots()
    ind = np.arange(len(groups))
    rects_s = []
    for i in range(len(labels)):
        number = ''.join([i for i in labels[i][:15] if i.isdigit()])
        labels[i] = number
    #mappings = {'2575':"minDist",'2275':"Pos,Vel,Angle", '2276':"Pos,Vel,Angle. Imaginary"}

    mappings = {}
    for i in range(len(labels)):
        if labels[i] in mappings.keys():
            labels[i] = mappings[labels[i]]

    for i, rect_data in enumerate(rects_data):
        r = ax.bar(ind + i*width, rect_data, width, label=labels[i])
        rects_s.append(r)

    ax.set_ylabel('Success rate best policy')
    ax.set_title('Rates with tollerance of N collisions')
    ax.set_xticks(ind)
    ticks_labels = []
    for g in groups:
        ticks_labels.append('N={}'.format(g))
    ax.set_xticklabels(ticks_labels)
    ax.legend(prop={'size': 8})


    plt.savefig(os.path.join(args.dir, 'fig_{}.pdf'.format(env_id)), format='pdf')
    if args.save_path:
        plt.savefig(args.save_path)
