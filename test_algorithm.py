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
            
experiment_name = 'test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
   
enemy = 1
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(),
                  enemies=[enemy])

def fitness(population):
    # TODO: Check fitness
    return [x is 0 for x in range(len(population))]

def offspring(solutions):
    # TODO: Create offsprings according to the algorithm
    # Find fitness only for the new population
    return solutions

pop_size = 30
gen = 20
n_hidden = 10

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5

# create initial population and add to environment
pop = np.random.uniform(-1, 1, (pop_size, n_vars))
pop_fitness = fitness(pop)

# TODO: SAVE BEST WEIGHT VALUES
best_each_gen = np.argmax(pop_fitness)
mean_each_gen = [np.mean(pop_fitness)]
std_each_gen = [np.std(pop_fitness)]

solutions = [pop, pop_fitness]
env.update_solutions(solutions)

for i in range(gen):
    solutions = offspring(solutions)
    pop, pop_fitness = solutions
    
    new_best = np.argmax(pop_fitness)
    if new_best > best_each_gen[-1]:
        best_each_gen.append(new_best)
        # TODO: Change best genome to new best
    else:
        best_each_gen.append(best_each_gen[-1])
    
    mean_each_gen.append(np.mean(pop_fitness))
    std_each_gen.append(np.std(pop_fitness))
    