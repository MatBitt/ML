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

    def feedfoward(self, x, sig=True):
        # y = wx + b
        res = np.dot(self.w, x) + self.b
        if(sig):
            return sigmoid(res)
        return res

class RedeNeural:
    def __init__(self, features):
        self.features = features

        wi = []
        bi = []
        wo = []

        for i in range(features**2):
            wi.append(np.random.normal())
        
        for i in range(features):
            wo.append(np.random.normal())
            bi.append(np.random.normal())
            
        self.wi = np.array(wi)
        self.wo = np.array(wo)
        self.bi = np.array(bi)            
        self.bo = np.random.normal()

    def feedfoward(self, x):
        #Input layer
        saida = []

        for j in range(self.features):
            featureset = []
            for i in range(self.features):
                featureset.append(self.w[j*self.features + i])
            n = Neuronio(np.array(featureset), self.bi[i])
            saida.append(n.feedfoward(x))


        #Output layer
        n_out = Neuronio(self.wo, self.bo)
        res = n_out.feedfoward(np.array(saida))

        return res
    
    def pesos(self):

        for i in range(self.features**2):
            print(f"wi{i} : {self.wi[i]}")

        for i in range(self.features):
            print(f"wo{i} : {self.wo[i]}")
        
        for i in range(self.features):
            print(f"bi{i} : {self.bi[i]}")
        
        print(f"bo0 : {self.bo}")

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

                saida = []
                res_n = []

                for j in range(self.features):
                    featureset = []
                    for i in range(self.features):
                        featureset.append(self.w[j*self.features + i])
                    n = Neuronio(np.array(featureset), self.bi[i])
                    res_n.append(n.feedfoward(x), sig=False)
                    saida.append(n.feedfoward(x))

                last_n = Neuronio(self.wo, self.bo)
                n_out = n_out.feedfoward(np.array(saida), sig=False)
                y_pred = sigmoid(n_out)

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
            

# Cada dado possui 2 features, [x1,x2]
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

rede = RedeNeural(2)
print('Pesos iniciais:')
rede.pesos()


# rede.train(data, y_reais)
# print('Pesos finais:')
# rede.pesos()


# Coisas a adicionar

# Achar um jeito bom de estimar os pesos iniciais
# Dividir o learning rate em 3 etapas, começando com passos grandes, e dps diminuindo
# Tornar o algoritmo mais geral, sem levar em conta o numero de features
# Criar um pdf explicando a teoria por tras
