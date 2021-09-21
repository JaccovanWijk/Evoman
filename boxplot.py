
from neat_test import N_runs
import neat
import pickle
import numpy as np
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from player_controllers import player_controller
from box_plot_test import boxplot

experiment_name="neat_nhidden10_gen20_enemy1" # Kan ook variabele zijn zodat we over meerdere tests kunnen loopen
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

# print(five_runs(3, experiment_name))
fitnesses = []
for i in range(N_runs):
    fitnesses.append(five_runs(i, experiment_name))
fitnesses = [[1,1,2],[1,2,3],[3,2,2]]
#np.save(f"{experiment_name}/boxplotfitness", fitnesses)
boxplot(fitnesses)
