import numpy as np
import matplotlib.pyplot as plt
import warnings
from math import sqrt
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dados = {'grupo 1' : [[1,2], [2,3], [3,1]], 'grupo 2' : [[6,5],[7,7],[8,6]]}
novo_dado = [3,3]

def k_nearest_neighbors(dados, novo_dado, k=3):
    if len(dados) >= k:
        warnings.warn('Conjunto de dados muito pequeno')

    distancia = []

    for grupo in dados:
        for ponto in dados[grupo]:
            distancia_euclidiana = np.linalg.norm(np.array(ponto)-np.array(novo_dado))
            distancia.append([distancia_euclidiana, grupo])
    
    proximidades = [i[1] for i in sorted(distancia)[:k]]
    maior_proximidade = Counter(proximidades).most_common(1)[0][0]

    return maior_proximidade

result = k_nearest_neighbors(dados, novo_dado)
print(result)