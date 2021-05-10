import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pso_lib as psol
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data


data = read_tsp('C:\\Users\\xxx\\Desktop\\Yapay Zeka Ödev\\Ödev Çalışmaları\\pso\\berlin52.tsp')

data = np.array(data)
data = data[:, 1:]
plt.subplot(2, 2, 1)
plt.title('Şehirlerin İlk dizilişi')
show_data = np.vstack([data, data[0]])
plt.plot(data[:, 0], data[:, 1])

pso = psol.PSO(num_city=data.shape[0],data=data.copy(),iterasyon=1000,population=50)
Best_path, Best = pso.run()
print(Best)
plt.subplot(1, 2, 2)

Best_path = np.vstack([Best_path, Best_path[0]])
plt.plot(Best_path[:, 0], Best_path[:, 1])
plt.title('Sonuç')
plt.show()