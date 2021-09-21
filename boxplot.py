
import neat
import pickle
import numpy as np
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from player_controllers import player_controller

experiment_name="neat"
n_hidden = 10
local_dir = os.path.dirname(__file__)
config_path = os.path.join (local_dir,'neat_config_file.txt')

env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden))


def replay_genome(config_path, genome_path=f"{experiment_name}/winner_2.pkl"):
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

def five_runs():
    f_r = []
    for i in range(0,5):
        fit = replay_genome(config_path)
        f_r.append(fit)
    return f_r

print(five_runs())
