import numpy as np

np.random.seed(8) # Saída sempre igual
randomico = np.random.randint(0, 5, 5) # Gera um iterador de tamanho 5 com aleatórios

array = np.array([[[3,6], [5,9], [6,9]], [[9,9], [11,5], [0, 8]]])
print(array.ndim)
print(array.shape)