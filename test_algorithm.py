#from __future__ import print_function
import sys, os

import random
from numpy.lib.shape_base import _expand_dims_dispatcher
sys.path.insert(0, 'evoman') 
import neat       
import pickle   
import numpy as np
from evoman.environment import Environment
from player_controllers import player_controller2
from plot import plot_fitness

### TODO: MOET ERUIT BIJ INLEVEREN
# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
            
pop_size = 50
gen = 50
n_hidden = 5
N_runs = 10
enemy = 6
keep_old = 0.2

experiment_name = f"crossover_nhidden5_gen{gen}_enemy{enemy}"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller2(n_hidden),
                  enemies=[enemy],
                  randomini="yes")

def fitness(population, i):
    pop_fitness = []
    for individual in population:
        #reshaped = np.reshape(individual, (1,len(individual)))
        fitness = env.play(pcont=individual)[0]
        pop_fitness.append(fitness)
    
    fitness_gens.append(np.mean(pop_fitness))       # adding mean fitness to list
    np.save(f"{experiment_name}/fitness_gens_{i}", fitness_gens)   # saving to numpy file, opening in test.py
    fitness_max.append(np.max(pop_fitness))         # adding max fitness to list
    np.save(f"{experiment_name}/fitness_max_{i}", fitness_max)     # saving to numpy file, opening in test.py
     
    return pop_fitness

def offspring(solutions): #, old):
    population, pop_fitness = solutions
    new_population = np.zeros((len(pop),len(pop[0])))
    
    # parents_staying = int(len(population) * old)
    # new_children = len(population) - parents_staying
    
    # # Add highest parents
    # #TODO: add hightest parents
    # top_index = sorted(range(len(pop_fitness)), key=lambda i: pop_fitness[i])[-parents_staying:]
    # for p, i in enumerate(top_index):
    #     new_population[p] = population[i]
    
    # get weights according to relative fitness
    if (min(pop_fitness) < 0):
        positive = [x + min(pop_fitness) for x in pop_fitness]
        pop_weights = [x/sum(positive) for x in positive]
    else:
        pop_weights = [x/sum(pop_fitness) for x in pop_fitness]
        
    
    # Make 100% new population with uniform crossover
    for c in range(population): #new_children):
        # #Ugly but works
        # c += parents_staying
        
        # Choose random parents with weights in mind
        parents = random.choices(population, weights=pop_weights, k=2)
        
        # Pick every gene of the parents randomly
        parent_length = len(parents[0])
        child = np.zeros(parent_length)
        for j in range(parent_length):
            gene = random.choice([parents[0][j], parents[1][j]])
            child[j] = gene
        new_population[c] = child

    return new_population

fitness_gens = []
fitness_max = []

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5

for r in range(N_runs):

    # create initial population and add to environment
    pop = np.random.uniform(-1, 1, (pop_size, n_vars))
    pop_fitness = fitness(pop, r)
    
    best_each_gen = [np.max(pop_fitness)]
    best = pop[np.argmax(pop_fitness)]
    mean_each_gen = [np.mean(pop_fitness)]
    std_each_gen = [np.std(pop_fitness)]
    
    print("\n------------------------------------------------------------------")
    print(f"Generation 0. Mean {mean_each_gen[-1]}, best {best_each_gen[-1]}")
    print("------------------------------------------------------------------")
    
    solutions = [pop, pop_fitness]
    env.update_solutions(solutions)
    
    for i in range(gen):
        pop = offspring(solutions, keep_old)
        pop_fitness = fitness(pop, r)
        
        new_best = np.max(pop_fitness)
        if new_best > best_each_gen[-1]:
            best = pop[np.argmax(pop_fitness)]
        best_each_gen.append(new_best)
        
        mean_each_gen.append(np.mean(pop_fitness))
        std_each_gen.append(np.std(pop_fitness))
         
        print("\n------------------------------------------------------------------")
        print(f"Generation {i+1}. Mean {mean_each_gen[-1]}, best {best_each_gen[-1]}")
        print("------------------------------------------------------------------")
        
        solutions = [pop, pop_fitness]
        env.update_solutions(solutions)
    
    # Stackoverflow on how to save the winning file and open it: https://stackoverflow.com/questions/61365668/applying-saved-neat-python-genome-to-test-environment-after-training
    with open(f"{experiment_name}/winner_{r}.pkl", "wb") as f:
        pickle.dump(best, f)
        f.close()