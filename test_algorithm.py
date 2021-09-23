#from __future__ import print_function
import sys, os

from numpy.lib.shape_base import _expand_dims_dispatcher
sys.path.insert(0, 'evoman') 
import neat       
import pickle   
import numpy as np
from evoman.environment import Environment
from player_controllers import player_controller2
from plot import plot_fitness

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
            
experiment_name = 'test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
   
enemy = 1
n_hidden = 10
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller2(n_hidden),
                  enemies=[enemy],
                  randomini="yes")

def fitness(population):
    pop_fitness = []
    for individual in population:
        reshaped = np.reshape(individual, (1,len(individual)))
        fitness = env.play(pcont=reshaped)[0]
        pop_fitness.append(fitness)
    
    fitness_gens.append(np.mean(pop_fitness))       # adding mean fitness to list
    np.save(f"{experiment_name}/fitness_gens_{i}", fitness_gens)   # saving to numpy file, opening in test.py
    fitness_max.append(np.max(pop_fitness))         # adding max fitness to list
    np.save(f"{experiment_name}/fitness_max_{i}", fitness_max)     # saving to numpy file, opening in test.py
     
    return pop_fitness

def offspring(solutions):
    # TODO: Create offsprings according to the algorithm
    # Find fitness only for the new population
    population, pop_fitness = solutions
    new_population = [[]]
    chances = (pop_fitness+10)/110 #normalize -10/100 to 0/1
    while len(new_population) < len(population):
        parents = np.random.choice(population, 2, p=chances) # Pick two parents based on their fitness
        child = parents[1][:len(parents[1])] + parents[2][len(parents[2]):]
        new_population[0].add(child)
    return new_population

fitness_gens = []
fitness_max = []

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
    pop = offspring(solutions)
    pop_fitness = fitness(pop)
    solutions = [pop, pop_fitness]
    env.update_solutions(solutions)
    
    new_best = np.argmax(pop_fitness)
    if new_best > best_each_gen[-1]:
        best_each_gen.append(new_best)
        # TODO: Change best genome to new best
    else:
        best_each_gen.append(best_each_gen[-1])
    
    mean_each_gen.append(np.mean(pop_fitness))
    std_each_gen.append(np.std(pop_fitness))

# TODO: Save winner