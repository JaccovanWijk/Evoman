import neat
import pickle
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0, 'evoman') 
from evoman.environment import Environment
from player_controllers import player_controller
import regex as re

def replay_genome(config_path, run_i, experiment_name):
    """
    Picks a winner.pkl for run_i and experiment_name and reruns the genome to get the gain.
    """
    genome_path = f"{experiment_name}/winner_{run_i}.pkl"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # playing genome and getting gain
    player_life, enemy_life = env.play(pcont=genome)[1:3]
    gain = player_life - enemy_life

    return gain

def five_runs(run_i, experiment_name):
    """
    Runs replay_genome() five times and saves the fitnesses.
    """
    f_r = []
    for i in range(0,5):
        gain = replay_genome(config_path, run_i, experiment_name)
        f_r.append(gain)
    return f_r


# used variables
N_runs = 10
local_dir = os.path.dirname(__file__)
config_path = os.path.join (local_dir,'neat_config_file.txt')
            
# reading all directories and saving all starting with experiment names with specified variables
directories = [name for name in os.listdir(".") if os.path.isdir(name)]
enemies = []
experiment_names = []

for dir in directories:
    if re.match("neat_nhidden5_gen50_enemy", dir):
        enemies.append(int(re.findall(r"enemy\d{1,2}", dir)[0][5:]))
        experiment_names.append(dir)
    if re.match("neat_sigma_nhidden5_gen50_enemy",dir):
        enemies.append(int(re.findall(r"enemy\d{1,2}", dir)[0][5:]))
        experiment_names.append(dir)

# sorting enemies (copied from https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list)
experiment_names = [x for _, x in sorted(zip(enemies, experiment_names))]
enemies.sort()

# saving all data for the boxplots
boxplotdata = []

# saving the mean gain from every five runs for every experiment name
for i, experiment_name in enumerate(experiment_names):
    gains = []
    env = Environment(experiment_name=experiment_name,
            playermode="ai",
            player_controller=player_controller(),
            enemies=[enemies[i]],
            randomini="yes")
    
    gains = [np.mean(five_runs(j, experiment_name)) for j in range(0, N_runs)]
    
    # saving the data in an array for plt.boxplot() in shape (enemy, gains)
    boxplotdata.append(gains)

# changing enemy names with sigma for xticks
for i, enemy in enumerate(enemies):
    if (i % 2) != 0:
        enemies[i] = f"{enemy} Sigma"

plt.figure()
plt.boxplot(boxplotdata)
plt.xticks(np.arange(0, len(enemies))+1, enemies)
plt.ylabel("individual gain")
plt.xlabel('Enemy')
plt.title('Individual Gain\nNormal vs Sigma Scaling')
plt.savefig(f"boxplotfigs/boxplot_randomini_neat_original", dpi=400)
plt.show()
