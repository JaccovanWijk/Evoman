
from neat_test import N_runs
import neat
import pickle
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0, 'evoman') 
from environment import Environment
from player_controllers import player_controller

# experiment_name="neat_nhidden10_gen20_enemy1" # Kan ook variabele zijn zodat we over meerdere tests kunnen loopen
N_runs = 10
n_hidden = 10
local_dir = os.path.dirname(__file__)
config_path = os.path.join (local_dir,'neat_config_file.txt')


def replay_genome(config_path, run_i, experiment_name, enemy):
    genome_path = f"{experiment_name}/winner_{run_i}.pkl"
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]
    env = Environment(experiment_name=experiment_name,
            playermode="ai",
            player_controller=player_controller(),
            enemies=[enemy])

    for genome_id, g in genomes:
        replay = env.play(pcont=g)
        gain = replay[1] - replay[2]
    return gain

def five_runs(run_i, experiment_name, enemy):
    f_r = []
    for i in range(0,5):
        gain = replay_genome(config_path, run_i, experiment_name, enemy)
        f_r.append(gain)
    return f_r

# reading all directories and saving all starting with "neat_nhidden10_gen20_enemy" and the enemies list
directories = [name for name in os.listdir(".") if os.path.isdir(name)]
enemies = []
experiment_names = []
for dir in directories:
    if dir[:26] == "neat_nhidden10_gen20_enemy":
        enemies.append(int(dir[-1]))
        experiment_names.append(dir)

# sorting the exp names and enemies alphabetically/numerically, so lists keep a corresponding order
experiment_names = np.sort(experiment_names)
enemies = np.sort(enemies)

# saving all data for the boxplots
boxplotdata = []
plt.figure()
for i, experiment_name in enumerate(experiment_names):
    # doing every run from N_runs 5 times and saving the mean
    gains = []
    for j in range(0, N_runs):
        #print(enemies[i])
        gains.append(np.mean(five_runs(j, experiment_name, enemies[i])))
    # saving the data in a 1D array for plt.boxplot
    boxplotdata.append(gains)
plt.boxplot(boxplotdata)
# getting the xlabels for correct enemies
plt.xticks(np.arange(0, len(enemies))+1, enemies)
plt.ylabel("individual gain")
plt.xlabel("enemy")
plt.show()