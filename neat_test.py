#from __future__ import print_function
import sys, os
sys.path.insert(0, 'evoman') 
import neat           
from environment import Environment
from player_controllers import player_controller
            
experiment_name = 'neat_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller())


def fitness_player(genomes, config):
    for genome_id, g in genomes:
        g.fitness = 0
        g.fitness = env.play(pcont=g)
        

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(fitness_player, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    
if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_config_file.txt')
    run(config_path)