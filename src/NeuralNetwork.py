import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #Normarlizar a saída para ser entre 0 e 1

def derivada_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1-sig)

def erro_quadratico(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

class Neuron:
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
        n1 = Neuron(np.array([self.w1, self.w2]), self.b1)
        saida_n1 = n1.feedfoward(x)
        n2 = Neuron(np.array([self.w3, self.w4]), self.b2)
        saida_n2 = n2.feedfoward(x)

        #Camada 2
        n3 = Neuron(np.array([self.w5, self.w6]), self.b3)
        saida_n3 = n3.feedfoward(np.array([saida_n1, saida_n2]))
        return saida_n3
    
    def train():
        pass

teste = RedeNeural()
features = np.array([2, 3])

print(teste.feedfoward(features))
