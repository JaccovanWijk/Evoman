#### Moved to boxplot.py

# import matplotlib.pyplot as plt
# import numpy as np

# def boxplot(fitnesses):
#     plt.figure()
#     x = np.arange(fitnesses.shape[0])
#     for i in x:
#         values = fitnesses[i,:]
#         mean = np.mean(values, axis=0)
#         stdev = np.std(values, axis=0)

#         data = np.concatenate((mean, mean+stdev, mean-stdev))

#         plt.boxplot(data)
#     plt.show()