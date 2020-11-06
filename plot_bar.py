import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    N = 4
    hgg_rates = (0., 0.01, 0.2, 0.78)
    bbox_rates = (0., 0.08, 0.58, 0.78)
    optimal_rates = (0., 0.11, 0.71, 0.96)

    ind = np.arange(N)
    width = 0.2
    plt.bar(ind, hgg_rates, width, label='HGG')
    plt.bar(ind + width, bbox_rates, width, label='Bbox')
    plt.bar(ind + 2*width, optimal_rates, width, label='Optimal')

    plt.ylabel('Success rate best policy')
    plt.title('Rates with tollerance of N collisions')

    plt.xticks(ind + width / 3, ('N=2', 'N=4', 'N=7', 'N=10'))
    plt.legend(loc='best')
    plt.savefig('comparison.png')
    plt.close()