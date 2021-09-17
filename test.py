import numpy as np
import matplotlib.pyplot as plt

fitness = np.load('fitness_gens.npy')
fitness_max = np.load('fitness_max.npy')

plt.figure()
plt.plot(fitness, 'bo')
plt.plot(fitness_max)
plt.show()