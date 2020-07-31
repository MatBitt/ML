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
                featureset.append(self.wi[j*self.features + i])
            n = Neuronio(np.array(featureset), self.bi[i])
            saida.append(n.feedfoward(x))


        #Output layer
        n_out = Neuronio(self.wo, self.bo)
        res = n_out.feedfoward(np.array(saida))

        return res
    
    def pesos(self):

        for i in range(self.features**2):
            print(f"wi{i} : {self.wi[i]:.3f}")

        for i in range(self.features):
            print(f"wo{i} : {self.wo[i]:.3f}")
        
        for i in range(self.features):
            print(f"bi{i} : {self.bi[i]:.3f}")
        
        print(f"bo0 : {self.bo:.3f}")

        print(' ')
    
    def train(self, data, y_reais):

        learn_rate = 0.1
        iteracoes = 1000

        for iteracao in range(iteracoes):
            for x, y_real in zip(data, y_reais):

                n = []
                res_n = []

                for j in range(self.features):
                    featureset = []
                    for i in range(self.features):
                        featureset.append(self.wi[j*self.features + i])
                    node = Neuronio(np.array(featureset), self.bi[i])
                    res_n.append(node.feedfoward(x, sig=False))
                    n.append(node.feedfoward(x))

                last_n = Neuronio(self.wo, self.bo)
                n_out = last_n.feedfoward(np.array(n), sig=False)
                y_pred = sigmoid(n_out)

                # Calculo das derivadas parciais
                # L denota loss, que é a função do erro
                # dL_dw1 significa "derivada parcial de L / derivada parcial de w1"

                dL_dypred = -2 * (y_real - y_pred)

                # Output node

                dypred_dw = []
                dypred_dn = []
                for i in range(self.features):
                    dypred_dw.append(n[i] * derivada_sigmoid(n_out))
                    dypred_dn.append(self.wo[i] * derivada_sigmoid(n_out))
                
                dypred_dbo = derivada_sigmoid(n_out)

                # Input nodes
                dn_dw = []
                aux = []

                for j in range(self.features):
                    for i in range(self.features):
                        aux.append(x[i] * derivada_sigmoid(res_n[j]))
                    dn_dw.append(aux)

                dn_db = []
                for i in range(self.features):
                    dn_db.append(derivada_sigmoid(res_n[i]))

                # Atualizando
                # x = x - (alfa * (dL/dx))

                # Input nodes
                for j in range(self.features):
                    for i in range(self.features):
                        self.wi[j*self.features + i] -= learn_rate * dL_dypred * dypred_dn[j] * dn_dw[j][i]
                
                for i in range(self.features):
                    self.bi[i] -= learn_rate * dL_dypred * dypred_dn[i] * dn_db[i]

                # Output node
                for i in range(self.features):
                    self.wo[i] -= learn_rate * dL_dypred * dypred_dw[i]
                
                self.bo -= learn_rate * dL_dypred * dypred_dbo

            if (iteracao % 10) == 0:
                y_preds = np.apply_along_axis(self.feedfoward, 1, data)
                erro = erro_quadratico(y_reais, y_preds)
                print("iteracao %d erro: %.3f" %(iteracao, erro))
                pass    


data = np.array([
  [-2, -1, 5],  # Dado 1
  [25, 6, 1],   # Dado 3
  [17, 4, 2],   # Dado 3
  [-15, -6, 7], # Dado 4
])

y_reais = np.array([
  1, # Dado 1
  0, # Dado 2
  0, # Dado 3
  1, # Dado 4
])

teste = np.array([20, 2, -1])

rede = RedeNeural(3)
# print('Pesos iniciais:')
# rede.pesos()

rede.train(data, y_reais)
# print('Pesos finais:')
# rede.pesos()

# print(rede.feedfoward(teste))


# Coisas a adicionar

# Achar um jeito bom de estimar os pesos iniciais
# Dividir o learning rate em 3 etapas, começando com passos grandes, e dps diminuindo
# Criar um pdf explicando a teoria por tras
