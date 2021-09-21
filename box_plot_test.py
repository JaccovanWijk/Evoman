import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,10)
values = np.random.random((10,100))
mean = np.mean(values, axis=1)
stdev = np.std(values, axis=1)

data = np.concatenate((mean, mean+stdev, mean-stdev))

plt.figure()
plt.boxplot(data)
plt.show()