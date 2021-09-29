
import neat
import pickle
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.insert(0, 'evoman') 
from environment import Environment
from player_controllers import player_controller
import regex as re

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# experiment_name="neat_nhidden10_gen20_enemy1" # Kan ook variabele zijn zodat we over meerdere tests kunnen loopen
N_runs = 10
n_hidden = 10
local_dir = os.path.dirname(__file__)
config_path = os.path.join (local_dir,'neat_config_file.txt')
enemy = 6


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
        replay = env.play(pcont=g)
        gain = replay[1] - replay[2]
    return gain

def five_runs(run_i, experiment_name):
    f_r = []
    for i in range(0,5):
        gain = replay_genome(config_path, run_i, experiment_name)
        f_r.append(gain)
    return f_r

# reading all directories and saving all starting with "neat_nhidden10_gen20_enemy" and the enemies list
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
experiment_names = [x for _, x in sorted(zip(enemies, experiment_names))]
enemies.sort()


# experiment_names.append("neat_nhidden5_gen20_randomini_enemy6")
# enemies.append("6_randomini")

    # # for randomini = yes
    # if dir[:len(experiment_name0)] == experiment_name0:
    #     enemies.append(int(dir[-1]))
    #     experiment_names.append(dir)
    # # for randomini = no
#     if dir[-16:-1] == "randomini_enemy":
#         enemies.append(int(dir[-1]))
#         experiment_names.append(dir)
# print(enemies)
# print(experiment_names)



# saving all data for the boxplots
boxplotdata = []
plt.figure()
for i, experiment_name in enumerate(experiment_names):
    # doing every run from N_runs 5 times and saving the mean
    gains = []
    env = Environment(experiment_name=experiment_name,
            playermode="ai",
            player_controller=player_controller(),
            enemies=[enemies[i]],
            randomini="yes")
    for j in range(0, N_runs):
        #print(enemies[i])
        gains.append(np.mean(five_runs(j, experiment_name)))#, enemies[i])))
    boxplotdata.append(gains)
    # saving the data in a 1D array for plt.boxplot
for i, enemy in enumerate(enemies):
    if (i % 2) != 0:
        enemies[i] = f"{enemy} Sigma"
plt.boxplot(boxplotdata)
# getting the xlabels for correct enemies
plt.xticks(np.arange(0, len(enemies))+1, enemies)
plt.ylabel("individual gain")
plt.xlabel('Enemy')
plt.title('Individual Gain Normal vs Sigma Scaling')
plt.savefig(f"boxplotfigs/boxplot_randomini_neat_original", dpi=400)
plt.show()