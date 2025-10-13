from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 300)
y = np.cos(3*x)

fig, axe = plt.subplots(figsize=(7, 4))
axe.plot(x, y)
axe.set_title("Gráfico do Cosseno", fontsize=16)
axe.set_xlabel("Eixo X", fontsize=14)
axe.set_ylabel("Eixo Y", fontsize=14)

# Manipulando escala dos eixos
plt.xticks(np.arange(0, 2*np.pi + 0.5, 0.5))
plt.yticks(np.arange(-1, 1.2, 0.2))


plt.grid(True)
plt.show()

