import numpy as np
import matplotlib.pyplot as plt
import warnings
from math import sqrt
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

data = {'grupo 1' : [[1,2], [2,3], [3,1]], 'grupo 2' : [[6,5],[7,7],[8,6]]}
novo_dado = [3,3]

def k_nearest_neighbors(data, novo_dado, k=3):
    if len(data) >= k:
        warnings.warn('Conjunto de dados muito pequeno')

    distancias = []

    for grupo in data:
        for ponto in data[grupo]:
            distancia = np.linalg.norm(np.array(ponto)-np.array(novo_dado))
            distancias.append([distancia, grupo])
    
    proximidades = [i[1] for i in sorted(distancia)[:k]]
    maior_proximidade = Counter(proximidades).most_common(1)[0][0]

    return maior_proximidade

result = k_nearest_neighbors(data, novo_dado)
print(result)