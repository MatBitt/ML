import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean

style.use('fivethirtyeight')

class regressao_linear:
    def __init__(self, entradas, saidas):
        self.a = (((mean(entradas) * mean(saidas)) - mean(entradas*saidas)) / ((mean(entradas)**2) - mean(entradas**2)))
        self.b = mean(saidas) - self.a*mean(entradas)
        self.xs = entradas
        self.ys = saidas
 
    def previsao(self, entrada):
        saida = (self.a*entrada) + self.b
        plt.scatter(entrada, saida, color='r')
        return saida

    def grafico(self):
        self.previsao = np.array([(self.a*x) + self.b for x in self.xs])
        plt.scatter(self.xs, self.ys, color='g')
        plt.plot(self.xs, self.previsao)
        return self.previsao

    def acuracy(self):
        y = sum((self.previsao - self.ys) ** 2)
        y_mean = sum((self.ys - mean(self.ys)) ** 2)
        return str("{:.2f}".format((1 - (y/y_mean))*100)) + '%'

    def plot(self):
        plt.show()



xs = np.array([1, 2, 3, 4, 5, 6])
ys = np.array([4, 7, 6, 8, 9, 11])

regressao = regressao_linear(xs, ys)
regressao.previsao(8)
regressao.grafico()
print(regressao.acuracy())
regressao.plot()


