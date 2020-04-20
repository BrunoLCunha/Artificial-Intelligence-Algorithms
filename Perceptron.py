import numpy as np
class Perceptron:

    def __init__(self):
        self.inputs = np.array([[1,1,1,1], [-1,1,-1,-1], [1, 1, 1, -1], [1, -1, -1, 1]])
        self.targets = np.array([1,1,-1,-1])
        self.learning_rate = 1
        self.weights = np.array([0,0,0,0])
        self.bias_weight = 0
        super().__init__()
    
    def train(self, epochs):
        for o in range (epochs):
            for aux,i in enumerate(self.inputs):
                ans = self.predict(i)
                if ans != self.targets[aux]:
                    for idx,val in enumerate(self.weights):            
                        self.weights[idx] += (self.targets[aux] - ans) * self.inputs[aux][idx] * self.learning_rate
                        self.bias_weight += (self.targets[aux] - ans) * self.learning_rate
        return self.weights

    def predict(self,inputs):
        sum = 0
        for indx,i in enumerate(inputs):
            sum += i * self.weights[indx] 
        sum += self.bias_weight
        return self.step(sum)

    def step(self, x):
        return 1 if x >= 0 else -1

P = Perceptron()
epochs = 10
weights = P.train(epochs)
inputs_t = np.array([[1,1,1,1], [-1,1,-1,-1], [1, 1, 1, -1], [1, -1, -1, 1]])
print("\nEntradas:")
print(inputs_t)
print("\nSaidas:")
for e in inputs_t:
    res = P.predict(e)
    print(res)
print("\nPesos obtidos no treinamento com " + str(epochs) + " epocas\n")
print(weights)
