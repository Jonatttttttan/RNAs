import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize

x = np.linspace(0, 2*np.pi, 500)
c = np.cos(x)
s = np.sin(x)

plt.figure("Gráficos cosenoidais", figsize=(8, 4))
# Configuração de espaçamento dos gráficos
plt.subplots_adjust(
    left = 0.12,
    right = 0.964,
    top = 0.9,
    bottom = 0.14,
    wspace = 0.438,
    hspace = 0.4
)

ax1 = plt.subplot(1, 2, 1)
ax1.set_title("Gráfico do cosseno")
ax1.set_xlabel("Eixo de tempo")
ax1.set_ylabel("Eixo da amplitude")
ax1.grid()
plt.plot(x, c)

ax2 = plt.subplot(1, 2, 2)
ax2.set_title("Gráfico do seno")
ax2.set_xlabel("Eixo de tempo")
ax2.set_ylabel("Eixo da amplitude")
ax2.grid()
plt.plot(x, s)

plt.show()



'''
ax3 = plt.subplot(2, 2, 3)
ax3.set_title("Gráfico do seno")
plt.plot(x, s)

ax4 = plt.subplot(2, 2, 4)
ax4.set_title("Gráfico do cosseno")
plt.plot(x, c)
'''




