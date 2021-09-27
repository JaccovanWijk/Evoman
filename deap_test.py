import random
import sys, os
import numpy as np
sys.path.insert(0, 'evoman') 
from deap import base, creator, tools
from evoman.controller import Controller
from evoman.environment import Environment
def sigmoid_activation(x):
	    return 1./(1.+np.exp(-x))
   
# implements controller structure for player
class player_controller_deap(Controller):
    def __init__(self, _n_hidden=0):
		# Number of hidden neurons
        self.n_hidden = [_n_hidden]
    
    def control(self, inputs, controller):
		# Normalises the input using min-max scaling
        inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))
        if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
            bias1 = controller[:self.n_hidden[0]].reshape(1,self.n_hidden[0])
			# Weights for the connections from the inputs to the hidden nodes
            weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0]
            weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs),self.n_hidden[0]))

			# Outputs activation first layer.
            output1 = sigmoid_activation(inputs.dot(weights1) + bias1)

			# Preparing the weights and biases from the controller of layer 2
            bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1,5)
            weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0],5))

			# Outputting activated second layer. Each entry in the output is an action
            output = sigmoid_activation(output1.dot(weights2)+ bias2)[0]
        else:
            bias = controller[:5].reshape(1, 5)
            weights = controller[5:].reshape((len(inputs), 5))
            
            output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        # Weight of each action
        if output[0] > 0.5:
            left = 1
        else:
            left = 0
            
        if output[1] > 0.5:
            right = 1
        else:
            right = 0
            
        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0
        if output[3] > 0.5:
                shoot = 1
        else:
                shoot = 0

        if output[4] > 0.5:
                release = 1
        else:
                release = 0

        return [left, right, jump, shoot, release]
    


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

IND_SIZE = 20
N_OUTPUT = 5
N_SENSORS = 20

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("weigths", tools.initRepeat, toolbox.attribute, 
                 random.random, N_OUTPUT*N_SENSORS)

toolbox.register("population", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=IND_SIZE)

def evaluate(individual):
    print(f"ind weights: {toolbox.weigths}")
    fitness = env.play(pcont=np.random.random(105))[0]
    return fitness

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

if __name__ == "__main__":
    pop = toolbox.population(n=5)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 20

    # create environment
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
                
    experiment_name = 'deap_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    env = Environment(experiment_name=experiment_name,
                    playermode="ai",
                    player_controller=player_controller_deap(),
                    enemies=[6],
                    randomini="yes")


    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    print(list(fitnesses))
    print(zip(pop, fitnesses))
    for ind, fit in zip(pop, fitnesses):
        print(ind)
        ind.fitness.values = fit
        print(f"ind: {ind}")
    exit()
    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    # return pop