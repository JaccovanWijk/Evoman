
from neat_test import N_runs
import neat
import pickle
import numpy as np
import sys, os
import matplotlib.pyplot as plt

sys.path.insert(0, 'evoman') 
from environment import Environment
from player_controllers import player_controller
from box_plot_test import boxplot

experiment_name="neat_nhidden10_gen20_enemy2"
N_runs = 10
n_hidden = 10

local_dir = os.path.dirname(__file__)
config_path = os.path.join (local_dir,'neat_config_file.txt')

env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller())


def replay_genome(config_path, run_i, experiment_name):
    genome_path = f"{experiment_name}/winner_{run_i}.pkl"
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    for genome_id, g in genomes:
        fitness = env.play(pcont=g)[0]
    return fitness

def five_runs(run_i, experiment_name):
    f_r = []
    for i in range(0,5):
        fit = replay_genome(config_path, run_i, experiment_name)
        f_r.append(fit)
    return f_r

enemies = [2,3]
plt.figure()
data = []
for e in enemies:
    experiment_name = experiment_name[:-1]+f"{e}"
    fitnesses = np.zeros((N_runs, 5))
    for i in range(N_runs):
        fitnesses[i,:] = five_runs(i, experiment_name)
    np.save(f"{experiment_name}/boxplotfitness", fitnesses)
    data.append(np.mean(fitnesses, axis=1))

plt.boxplot(data)
plt.xticks(np.arange(len(enemies))+1, enemies)
plt.show()