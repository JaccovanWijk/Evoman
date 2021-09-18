# from matplotlib.lines import _LineStyle
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, 'evoman') 

def plot_fitness(experiment_name, N_runs):

    local_dir = os.path.dirname(__file__)

    plt.figure()
    for i in range(N_runs):
        f_mean = np.load(f"{experiment_name}/fitness_gens_{i}.npy")
        f_max = np.load(f"{experiment_name}/fitness_max_{i}.npy")

        lines = []
        lines.append(plt.plot(f_mean, '-')[0])
        lines.append(plt.plot(f_max, '--')[0])

    plt.xlabel('generations')
    plt.ylabel('fitness')
    plt.legend(lines, ['mean', 'max'])
    plt.show()


if __name__ == '__main__':
    experiment_name = 'neat'
    N_runs = 10

    plot_fitness(experiment_name, N_runs)
