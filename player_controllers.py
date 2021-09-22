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