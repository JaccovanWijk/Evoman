# from matplotlib.lines import _LineStyle
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import regex as re

from numpy.core.fromnumeric import mean, std
sys.path.insert(0, 'evoman') 

def plot_fitness(general_names, N_runs, gens=20):

    local_dir = os.path.dirname(__file__)

    directories = [name for name in os.listdir(".") if os.path.isdir(name)]
    enemies = []
    experiment_names = []

    for dir in directories:
        for general_name in general_names:
            if re.match(general_name, dir):
                enemies.append(int(re.findall(r"enemy\d{1,2}", dir)[0][5:]))
                experiment_names.append(dir)
                
    fitnesses = np.zeros((len(experiment_names), N_runs, 2, gens)) # different runs, mean and max, maximum 100 generations
            
    for exp_id, experiment_name in enumerate(experiment_names):       
        # plt.figure()
        # plt.title(f"{experiment_name}")
        for i in range(N_runs):
            f_mean = np.load(f"{experiment_name}/fitness_gens_{i}.npy")
            f_max = np.load(f"{experiment_name}/fitness_max_{i}.npy")
            fitnesses[exp_id, i, :, :len(f_mean)] = np.array((f_mean, f_max))
            # lines = []
            # lines.append(plt.plot(f_mean, '-')[0])
            # lines.append(plt.plot(f_max, '--')[0])

    # plt.xlabel('generations')
    # plt.ylabel('fitness')
    # plt.legend(lines, ['mean', 'max'])
    
    for i, experiment_name in enumerate(experiment_names):
        plt.figure(i%3)
        
        fitnesses[fitnesses==0] = np.nan
        mean_mean_fitness = np.nanmean(fitnesses[i,:,0,:], axis=0)
        stdev_mean_fitness = np.nanstd(fitnesses[i,:,0,:], axis=0)
        mean_mean_fitness = mean_mean_fitness[mean_mean_fitness != 0]
        
        if (i < len(experiment_names)/2):
            color = 'red'
            label = 'normal fitness'
        else:
            color = 'blue'
            label = 'sigma scaled'
        plt.plot(mean_mean_fitness, '-', label=label, color=color)
        plt.fill_between(np.arange(0, len(mean_mean_fitness)), mean_mean_fitness - stdev_mean_fitness, mean_mean_fitness + stdev_mean_fitness,
                        color=color, alpha=0.2)
        plt.title(f"normal fitness vs. sigma - enemy {experiment_name[-1]}")
        # plt.ylim(-9,80)
        plt.xlabel("generations")
        plt.ylabel("fitness")
        plt.legend(loc=4)
        plt.ylim(-9, 80)
        
        plt.savefig(f"meanfigs/neat_mean_enemy{experiment_name[-1]}", dpi=400)
        
    plt.show()


if __name__ == '__main__':
    experiment_names = ['neat_sigma_nhidden5_gen50_enemy', 'neat_nhidden5_gen50_enemy']
    N_runs = 10

    plot_fitness(experiment_names, N_runs, gens=50)
