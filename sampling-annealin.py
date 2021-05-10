import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
import sys


file_name = sys.argv[1] if len(sys.argv) > 1 else 'C:\\Users\\xxx\\Desktop\\Yapay Zeka Ödev\\Ödev Çalışmaları\\AntColony\\berlin52.txt'
nodelist = np.loadtxt(file_name, delimiter=',')
num_points = nodelist.shape[0]

distance_matrix = spatial.distance.cdist(nodelist, nodelist, metric='euclidean')

def cal_total_distance(routine):
    
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])



from sko.SA import SA_TSP

sa_tsp = SA_TSP(func=cal_total_distance, x0=range(num_points), T_max=100, T_min=1, L=1 * num_points)

best_points, best_distance = sa_tsp.run()

fig, ax = plt.subplots(2, 1)

best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = nodelist[best_points_, :]
ax[0].plot(sa_tsp.best_y_history)
ax[0].set_xlabel("İterasyon Sayısı")
ax[0].set_ylabel("Mesafe")
ax[1].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1],
           marker='o', markerfacecolor='b', color='c', linestyle='-')

ax[1].set_xlabel("X")
ax[1].set_ylabel("Y")
plt.show()