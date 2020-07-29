import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #Normarlizar a saída para ser entre 0 e 1

def derivada_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1-sig)

def erro_quadratico(y_real, y_pred):
    return ((y_real - y_pred)**2).mean()

class Neuronio:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def feedfoward(self, x):
        # y = wx + b
        res = np.dot(self.w, x) + self.b
        return sigmoid(res)

class RedeNeural:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
    
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedfoward(self, x):
        #Camada 1
        n1 = Neuronio(np.array([self.w1, self.w2]), self.b1)
        saida_n1 = n1.feedfoward(x)
        n2 = Neuronio(np.array([self.w3, self.w4]), self.b2)
        saida_n2 = n2.feedfoward(x)

        #Camada 2
        n3 = Neuronio(np.array([self.w5, self.w6]), self.b3)
        saida_n3 = n3.feedfoward(np.array([saida_n1, saida_n2]))

        return saida_n3
    
    def pesos(self):
        print('w1 : %.3f' %(self.w1))
        print('w2 : %.3f' %(self.w2))
        print('w3 : %.3f' %(self.w3))
        print('w4 : %.3f' %(self.w4))
        print('w5 : %.3f' %(self.w5))
        print('w6 : %.3f' %(self.w6))
        print('b1 : %.3f' %(self.b1))
        print('b2 : %.3f' %(self.b2))
        print('b3 : %.3f' %(self.b3))
        print(' ')
    
    def train(self, data, y_reais):

        learn_rate = 0.1
        iteracoes = 1000

        for iteracao in range(iteracoes):
            for x, y_real in zip(data, y_reais):

                # Feedfoward
                res_n1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                n1 = sigmoid(res_n1)

                res_n2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                n2 = sigmoid(res_n2)

                res_n3 = self.w5 * n1 + self.w6 * n2 + self.b3
                n3 = sigmoid(res_n3)
                y_pred = n3

                # Calculo das derivadas parciais
                # L denota Loss, que é a função do erro
                # dL_dw1 significa "derivada parcial de L / derivada parcial de w1"

                dL_dypred = -2 * (y_real - y_pred)

                # Neuronio n3
                dypred_dw5 = n1 * derivada_sigmoid(res_n3)
                dypred_dw6 = n2 * derivada_sigmoid(res_n3)
                dypred_db3 = derivada_sigmoid(res_n3)

                dypred_dn1 = self.w5 * derivada_sigmoid(res_n3)
                dypred_dn2 = self.w6 * derivada_sigmoid(res_n3)

                # Neuronio n1
                dn1_dw1 = x[0] * derivada_sigmoid(res_n1)
                dn1_dw2 = x[1] * derivada_sigmoid(res_n1)
                dn1_db1 = derivada_sigmoid(res_n1)

                # Neuronio n2
                dn2_dw3 = x[0] * derivada_sigmoid(res_n2)
                dn2_dw4 = x[1] * derivada_sigmoid(res_n2)
                dn2_db2 = derivada_sigmoid(res_n2)

                # Atualizando
                # var = var - (alfa * (dL/dvar))
                
                # Neuronio n1
                self.w1 -= learn_rate * dL_dypred * dypred_dn1 * dn1_dw1
                self.w2 -= learn_rate * dL_dypred * dypred_dn1 * dn1_dw2
                self.b1 -= learn_rate * dL_dypred * dypred_dn1 * dn1_db1

                # Neuronio n2
                self.w3 -= learn_rate * dL_dypred * dypred_dn2 * dn2_dw3
                self.w4 -= learn_rate * dL_dypred * dypred_dn2 * dn2_dw4
                self.b2 -= learn_rate * dL_dypred * dypred_dn2 * dn2_db2

                # Neuronio n3
                self.w5 -= learn_rate * dL_dypred * dypred_dw5
                self.w6 -= learn_rate * dL_dypred * dypred_dw6
                self.b3 -= learn_rate * dL_dypred * dypred_db3

            if (iteracao % 10) == 0:
                y_preds = np.apply_along_axis(self.feedfoward, 1, data)
                erro = erro_quadratico(y_reais, y_preds)
                print("iteracao %d erro: %.3f" %(iteracao, erro))
                pass    
            

# Cada dado possui 2 features, [x,y]
# Baseado em suas features, obtemos a saída 0 ou 1

data = np.array([
  [-2, -1],  # Dado 1
  [25, 6],   # Dado 3
  [17, 4],   # Dado 3
  [-15, -6], # Dado 4
])

y_reais = np.array([
  1, # Dado 1
  0, # Dado 2
  0, # Dado 3
  1, # Dado 4
])

rede = RedeNeural()
# print('Pesos iniciais:')
# rede.pesos()


rede.train(data, y_reais)
# print('Pesos finais:')
# rede.pesos()


# Coisas a adicionar

# Achar um jeito bom de estimar os pesos iniciais
# Dividir o learning rate em 3 etapas, começando com passos grandes, e dps diminuindo
# Tornar o algoritmo mais geral, sem levar em conta o numero de features
# Criar um pdf explicando a teoria por tras
