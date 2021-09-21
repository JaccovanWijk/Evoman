# from matplotlib.lines import _LineStyle
import numpy as np
import matplotlib.pyplot as plt
import sys, os

from numpy.core.fromnumeric import mean, std
sys.path.insert(0, 'evoman') 

def plot_fitness(experiment_name, N_runs, gens=20):

    local_dir = os.path.dirname(__file__)
    fitnesses = np.zeros((N_runs, 2, gens)) # different runs, mean and max, maximum 100 generations

    plt.figure()
    for i in range(N_runs):
        f_mean = np.load(f"{experiment_name}/fitness_gens_{i}.npy")
        f_max = np.load(f"{experiment_name}/fitness_max_{i}.npy")
        fitnesses[i, :, :len(f_mean)] = np.array((f_mean, f_max))
        lines = []
        lines.append(plt.plot(f_mean, '-')[0])
        lines.append(plt.plot(f_max, '--')[0])

    plt.xlabel('generations')
    plt.ylabel('fitness')
    plt.legend(lines, ['mean', 'max'])
    # plt.show()

    fitnesses[fitnesses==0] = np.nan
    mean_mean_fitness = np.nanmean(fitnesses[:,0,:], axis=0)
    stdev_mean_fitness = np.nanstd(fitnesses[:,0,:], axis=0)
    mean_mean_fitness = mean_mean_fitness[mean_mean_fitness != 0]

    plt.figure()
    plt.plot(mean_mean_fitness, 'r-', label="mean")
    plt.fill_between(np.arange(0, len(mean_mean_fitness)), mean_mean_fitness - stdev_mean_fitness, mean_mean_fitness + stdev_mean_fitness,
                    color='red', alpha=0.2, label="stdev")
    plt.xlabel("generations")
    plt.ylabel("fitness")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    experiment_name = 'neat'
    N_runs = 10

    plot_fitness(experiment_name, N_runs)
