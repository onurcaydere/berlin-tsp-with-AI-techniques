import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import sys
import pandas as pd



file_name = sys.argv[1] if len(sys.argv) > 1 else 'C:\\Users\\xxx\\Desktop\\Yapay Zeka Ödev\\Ödev Çalışmaları\\AntColony\\berlin52.txt'
nodelist = np.loadtxt(file_name, delimiter=',')
num_points = nodelist.shape[0]


distance_matrix = spatial.distance.cdist(nodelist, nodelist, metric='euclidean')

def cal_total_distance(routine):
   
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
from sko.ACA import ACA_TSP

aca = ACA_TSP(func=cal_total_distance, n_dim=num_points,
              size_pop=100, max_iter=700,
              distance_matrix=distance_matrix)

best_x, best_y = aca.run()

# %% Plot
fig, ax = plt.subplots(2,1)
best_points_ = np.concatenate([best_x, [best_x[0]]])
best_points_coordinate = nodelist[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r',markerfacecolor="b")
ax[0].plot(best_points_coordinate[0,0],best_points_coordinate[0,1],"o-r",markerfacecolor="r",marker=">")#Başlangıç ve bitiş yerim
pd.DataFrame(aca.y_best_history).cummin().plot(ax=ax[1])
plt.show()
