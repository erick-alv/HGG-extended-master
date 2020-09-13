import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()
import glob2
import argparse
from plot import smooth_reward_curve, load_results, pad

if __name__ == "__main__":
    '''
    --results_keys RealDirectDistance --name_suffix real_direct
    --results_keys LatentDirectDistance --name_suffix latent_direct
    --results_keys RealPathDistance --name_suffix real_path
    --results_keys LatentPathDistance --name_suffix latent_path
    --results_keys RealDirectToPrevDistance --name_suffix real_prev
    --results_keys LatentDirectToPrevDistance --name_suffix latent_prev
    '''
    # call plot.py to plot success stored in progress.csv files

    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('env_id', type=str)
    parser.add_argument('csv_file_name', type=str)

    parser.add_argument('--results_keys', nargs='+', default=[])
    parser.add_argument('--name_suffix', type=str, default=None)
    parser.add_argument('--smooth', type=int, default=1)
    parser.add_argument('--naming', type=int, default=0)
    parser.add_argument('--e_per_c', type=int, default=100)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()
    env_id = args.env_id

    # Load all data.
    data = {}
    path = os.path.abspath(os.path.join(glob2.glob(os.path.join(args.dir, '**', args.csv_file_name))[0],'..'))
    location = 2
    clean_path = path.replace(env_id, '')
    clean_path = os.path.basename(os.path.normpath(clean_path))
    clean_path = ''.join([i for i in clean_path if not i.isdigit()])
    # divide path into run (number in the beginning) and config (information on configuration, included in the path name)

    results = load_results(os.path.join(path, args.csv_file_name))

    # Filter out success rates from results
    for key, value in results.items():
        if key in args.results_keys:
            success_rate = np.array(results[key])
            iteration = (np.array(results['Step']))

            # Process and smooth data.
            assert success_rate.shape == iteration.shape
            x = iteration
            y = success_rate
            if args.smooth:
                x, y = smooth_reward_curve(iteration, success_rate)
            assert x.shape == y.shape

            # store everything in an array

            data[key] = []
            data[key].append((x, y))


    # Plot data.
    print('exporting {}'.format(env_id))
    plt.clf()

    # new curve for each config
    if args.naming == 4 or args.naming == 5 or args.naming == 1:
        configs = sorted(data.keys(), key=len)
    else:
        configs = sorted(data.keys())

    for config in configs:
        print("Config: {}".format(config))
        # merge curves from runs of one config
        xs, ys = zip(*data[config])
        xs, ys = pad(xs), pad(ys)
        assert xs.shape == ys.shape
        plt.plot(xs[0], np.nanmedian(ys, axis=0), label=config)
        plt.fill_between(xs[0], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25)

    plt.title(env_id)
    plt.xlabel('Step')
    #plt.ylabel('Median Success Rate')
    plt.legend(loc=location)

    plot_filename = args.csv_file_name[:-4]
    if args.name_suffix is not None:
        plot_filename += args.name_suffix
    else:
        suffix = '_'.join(args.results_keys)
        plot_filename += suffix
    plt.savefig(os.path.join(args.dir, '{}.pdf'.format(plot_filename)), format='pdf')
    if args.save_path:
        plt.savefig(args.save_path)