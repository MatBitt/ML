import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

class RedeNeural:

    def __init__(self, X, Y):
        X = X.T
        self.n_x = X.shape[0]
        self.n_h = 4
        self.n_o = Y.shape[0]

    def initializeParameters(self):
        self.W1 = np.random.randn(self.n_h, self.n_x)
        self.b1 = np.zeros((self.n_h, 1))
        self.W2 = np.random.randn(1, self.n_h)
        self.b2 = np.zeros((self.n_o, 1))
    
    def fowardPropagation(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = sigmoid(Z2)

        output = {"Z1":Z1,
                  "A1":A1,
                  "Z2":Z2,
                  "A2":A2}

        return output

    def backPropagation(self, X, Y, output):
        Z1 = output["Z1"]
        A1 = output["A1"]
        Z2 = output["Z2"]
        A2 = output["A2"]
        m = X.shape[1]

        dZ2 = (A2 - Y)
        dW2 = np.dot(dZ2, A1.T)/m
        db2 = np.sum(dZ2, axis = 1, keepdims = True)/m
        dZ1 = self.W2.T*dZ2*(1 - np.power(A1, 2))
        dW1 = np.dot(dZ1,X.T)/m
        db1 = np.sum(dZ1, axis = 1, keepdims = True)/m

        derivs = {"dW2":dW2,
                  "db2":db2,
                  "dW1":dW1,
                  "db1":db1
        }
    
        return derivs

    def updateParameters(self, derivs, learning_rate=0.1):
        dW1 = derivs["dW1"]
        db1 = derivs["db1"]
        dW2 = derivs["dW2"]
        db2 = derivs["db2"]

        self.W1 = self.W1 - learning_rate*dW1
        self.b1 = self.b1 - learning_rate*db1
        self.W2 = self.W2 - learning_rate*dW2
        self.b2 = self.b2 - learning_rate*db2

    def error(self, output, Y):
        A2 = output["A2"]
        m = Y.shape[0]

        log = -(np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y))/m
        cost = np.sum(log)
        
        return cost
    
    def predict(self, X):
        output = fowardPropagation(X)
        prediction = output["A2"]

        return prediction

    def model(self, X, Y, printError=False):
        X = X.T
        self.initializeParameters()
        iterations = 1000

        for i in range(iterations):
            output = self.fowardPropagation(X)
            derivs = self.backPropagation(X, Y, output)
            self.updateParameters(derivs)
            if printError and i%10 == 0:
                print(f"iteration {i} error :","%.3f" %self.error(output, Y))

data = np.array([
  [-2, -1, 5],  # Dado 1
  [25, 6, 1],   # Dado 2
  [17, 4, 2],   # Dado 3
  [-15, -6, 7], # Dado 4
])

y_reais = np.array([
  1, # Dado 1
  0, # Dado 2
  0, # Dado 3
  1, # Dado 4
])

teste = RedeNeural(data, y_reais)
teste.model(data, y_reais, printError=True)
