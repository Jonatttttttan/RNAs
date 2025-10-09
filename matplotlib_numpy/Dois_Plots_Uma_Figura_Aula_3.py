import matplotlib.pyplot as plt
import numpy as np

# t = np.arange(0, 100) # Valores de 0 a 99 inteiros
t = np.linspace(-np.pi, np.pi, 100)
y = np.cos(t)
y1 = np.sin(t)

# Dois gráficos em uma única figura
plt.figure("Gráfico", figsize=(6, 4))
plt.title("Gráficos do seno e cosseno")
plt.xlabel("Eixo de tempo")
plt.ylabel("Eixo da amplitude")
plt.plot(t, y) # Cosseno
plt.plot(t, y1) # Seno
plt.grid() # grade
plt.show()

