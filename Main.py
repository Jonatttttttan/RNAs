import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sklearn.datasets as datasets # dados online para treinamento

plt.rcParams["figure.figsize"] = (10,6)
plt.style.use("dark_background")

# Dataset
x,y = datasets.make_moons(n_samples = 500, noise = 0.05)
print(f'{x.shape =}, {y.shape = }')

#print(pd.DataFrame({'x_1' : x[:, 0], 'x_2' : x[:, 1], 'y' : y}))
# O dataframe acima indica que a variável x é uma tabela com 2 colunas e 500 linhas com valores arbitrários e y é a saída booleana.

plt.scatter(x[:, 0], x[:, 1], c = y, s = 50, alpha = 0.5, cmap = 'cool') # Mostra a distribuição dos dados
plt.show()

# Modelo
# - Inicialização dos pesos e bias
# - feedfoward
# - Backpropagation
# - Fit

class NnModel:
    def __init__(self, x: np.ndarray, y: np.ndarray, hidden_neurons: int = 10, output_neurons: int = 2):
        np.random.seed(8)
        self.x = x
        self.y = y
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.input_neurons = self.x.shape[1] # Número de colunas

        # Inicializa os pesos e bias
        # Xavier Inicialization -> variância dos pesos igual em todas as camadas

        self.W1 = np.random.randn(self.input_neurons, self.hidden_neurons) / np.sqrt(self.input_neurons)
        self.B1 = np.zeros((1, self.hidden_neurons))
        self.W2 = np.random.randn(self.hidden_neurons, self.output_neurons) / np.sqrt(self.hidden_neurons)
        self.B2 = np.zeros((1, self.output_neurons))
        self.model_dict = {'W1' : self.W1, 'B1' : self.B1, 'W2': self.W2, 'B2' : self.B2}
        self.z1 = 0
        self.f1 = 0


    def forward(self, x: np.ndarray) -> np.ndarray:
        # Equação da reta
        self.z1 = x.dot(self.W1) + self.B1

        # Função de ativação (1)
        self.f1 = np.tanh(self.z1) # Tangente hiperbólica

        # Equação da reta  (2)
        z2 = self.f1.dot(self.W2) + self.B2

        # função de ativação (2)
        f2 = np.tanh(z2)

        # Softmax - probabilidades das classes
        exp_values = np.exp(f2) # Exponencial e
        softmax = exp_values/np.sum(exp_values, axis = 1, keepdims=True) # axis =  coluna e keepdims mantém a dimensão
        return softmax

    def loss(self):
        pass

    def backpropagation(self):
        pass

    def fit(self):
        pass


modelo = NnModel(x, y, 10, 2)
print(modelo.forward(np.array([0.1, - 0.05])))
