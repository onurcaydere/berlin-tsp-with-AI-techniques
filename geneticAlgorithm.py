import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
import sys


file_name = sys.argv[1] if len(sys.argv) > 1 else 'C:\\Users\\xxx\\Desktop\\Yapay Zeka Ödev\\Ödev Çalışmaları\\AntColony\\berlin52.txt'
nodelist = np.loadtxt(file_name, delimiter=',')
num_points = nodelist.shape[0]

distance_matrix = spatial.distance.cdist(nodelist, nodelist, metric='euclidean')

def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# %% do GA

from sko.GA import GA_TSP
#iterasyon,populasyon, sayılarını değiştirerek iyi sonuçlar üretebiliriz.
ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=700, prob_mut=0.2)#n_dim kromozom uzunluğu
#prob_mu ile mutasyon oranını belirsem
#mutasyon oranımın yüksekliği gideceğim mesafeyi düşürür.
#TOPLAM 7971 MESAFE GİTMEKTEDİR.
#
best_points, best_distance = ga_tsp.run()

fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_points, [best_points[0]]])
print(best_points)
best_points_coordinate = nodelist[best_points_, :]

ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r',markerfacecolor="b")
ax[0].plot(best_points_coordinate[0,0],best_points_coordinate[0,1],"o-r",markerfacecolor="r",marker=">")#Başlangıç ve bitiş yerim


ax[1].plot(ga_tsp.generation_best_Y)
plt.show()
print(best_distance)
