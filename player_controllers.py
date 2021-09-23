from controller import Controller
import numpy as np
import os,neat

class player_controller(Controller):
    def __init__(self):
        # Initialize config by directing to config file
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'neat_config_file.txt')
        self.config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
									neat.DefaultSpeciesSet, neat.DefaultStagnation,
									config_path)
        
    def control(self, inputs, genome):
        # Creating and activating neural network using the genome and config
       net = neat.nn.FeedForwardNetwork.create(genome, self.config)
       output = net.activate(inputs)
       
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
   

def sigmoid_activation(x):
	return 1./(1.+np.exp(-x))
   
# implements controller structure for player
class player_controller2(Controller):
	def __init__(self, _n_hidden):
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

		# takes decisions about sprite actions
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