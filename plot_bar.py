import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

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

    ind = np.arange(N)
    width = 0.1
    show_results = 3
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


    plt.ylabel('Success rate best policy')
    plt.title('Rates with tollerance of N collisions')
    plt.xticks(ind + width / 3, ('N=2', 'N=4', 'N=7', 'N=10'))
    plt.legend(loc=4, prop={'size': 8})
    plt.savefig(figname)
    plt.close()