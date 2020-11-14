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
    optimal_rates = (0.02, 0.15, 0.68, 0.96)
    optimal_self_v1_rates = (0.04, 0.14, 0.72, 0.87)
    optimal_self_v2_rates = (0.11, 0.85, 0.95, 0.97)

    ind = np.arange(N)
    width = 0.1
    plt.bar(ind, hgg_rates, width, label='HGG')
    plt.bar(ind + width, bbox_rates, width, label='Bbox')
    #plt.bar(ind + 2 * width, optimal_self_v1_rates, width, label='Bbox multi-obj version1')
    plt.bar(ind + 3 * width, optimal_rates, width, label='Optimal')
    plt.bar(ind + 4 * width, optimal_self_v1_rates, width, label='Opt. multi-obj version1')
    plt.bar(ind + 5 * width, optimal_self_v2_rates, width, label='Opt. multi-obj version2')

    plt.ylabel('Success rate best policy')
    plt.title('Rates with tollerance of N collisions')

    plt.xticks(ind + width / 3, ('N=2', 'N=4', 'N=7', 'N=10'))
    plt.legend(loc='best')
    plt.savefig('comparison_with_no_velchange.png')
    plt.close()