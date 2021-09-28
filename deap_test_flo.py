import random
import sys, os
from numpy.lib.shape_base import _expand_dims_dispatcher
sys.path.insert(0, 'evoman')       
import pickle   
import numpy as np
from evoman.environment import Environment
from demo_controller import player_controller
from plot import plot_fitness
from deap import creator, base, tools, algorithms
toolbox = base.Toolbox()
pop_size = 100
n_hidden=5

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
            
experiment_name = 'Deap_testLog'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=player_controller(n_hidden),
                  enemies=[6],
                  randomini="yes")


# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5

# create initial population and add to environment
pop = np.random.uniform(-1, 1, (pop_size, n_vars))
#pop_fitness = fitness(pop)

def pop_fitness(population):
    pop_fit=[]
    for individual in population:
        reshaped = np.reshape(individual,(1,len(individual)))
        individualfit = env.play(pcont=reshaped[0])
        pop_fit.append(individualfit[0])
    return pop_fit


toolbox.register("evaluate", pop_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

gens = 20
population = pop_fitness(pop)

#for N in range(gens):
    #offspring = algorithms.varAnd(pop_fitness(pop), toolbox, cxpb=0.5, mutpb=0.1)
    #print(offspring)


for g in range(gens):
    population = select(pop, len(pop))
    offspring = varAnd(pop, toolbox, cxpb, mutpb)
    evaluate(offspring)
    population = offspring
print(population)


#print(pop_fitness(pop))

#creator.create("Individual", list, fitness=env.solutions[0])
#toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#pop = toolbox.population(n=100)

#print(pop)
#toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#print(individual)