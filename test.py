import numpy as np
import matplotlib.pyplot as plt

fitness = np.load('fitness_gens.npy')

print(fitness)

plt.figure()
plt.plot(fitness, 'bo')
plt.show()