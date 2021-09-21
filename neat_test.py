#from __future__ import print_function
import sys, os

from numpy.lib.shape_base import _expand_dims_dispatcher
sys.path.insert(0, 'evoman') 
import neat       
import pickle   
import numpy as np
from evoman.environment import Environment
from player_controllers import player_controller
from plot import plot_fitness

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
            
experiment_name = 'neat_pop30'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller())


def fitness_player(genomes, config):
    global i
    f_g = []
    for genome_id, g in genomes:
        # g.fitness = 0
        g.fitness = env.play(pcont=g)[0]
        f_g.append(g.fitness)
    fitness_gens.append(np.mean(f_g))       # adding mean fitness to list
    np.save(f"{experiment_name}/fitness_gens_{i}", fitness_gens)   # saving to numpy file, opening in test.py
    fitness_max.append(np.max(f_g))         # adding max fitness to list
    np.save(f"{experiment_name}/fitness_max_{i}", fitness_max)     # saving to numpy file, opening in test.py
        

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome,neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1000))

    # Run for up to 300 generations.
    winner = p.run(fitness_player, 20)
    
    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Stackoverflow on how to save the winning file and open it: https://stackoverflow.com/questions/61365668/applying-saved-neat-python-genome-to-test-environment-after-training
    with open(f"{experiment_name}/winner_{i}.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()


N_runs = 10

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config_file.txt')
    for i in range(N_runs):
        if not os.path.exists(f"{experiment_name}/winner_{i}.pkl"):
            fitness_gens = []       # list for mean fitness per generation
            fitness_max = []        # list for mean fitness per generation
            run(config_path)

    plot_fitness(experiment_name, N_runs)