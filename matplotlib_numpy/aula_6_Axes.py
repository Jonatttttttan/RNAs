import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize

x = np.arange(0, 5, 0.1) # 0, 0.1, 0.2, ..., 4.9
y1 = x**2
y2 = x**5
# subplots
fig, axes  = plt.subplots(nrows = 2, ncols = 2, figsize=(8, 4))
plt.suptitle("Gráficos com Subplots")
plt.subplots_adjust(
    left = 0.093,
    right = 0.948,
    top = 0.8,
    bottom = 0.148,
    wspace = 0.384,
    hspace = 0.824
)

axes[0, 0].plot(x, y1)
axes[0, 0].set_title("Exponencial x²")
axes[0, 0].set_xlabel("Eixo de tempo")
axes[0, 0].set_ylabel("Eixo da amplitude")

axes[1, 0].plot(x, y2)
axes[1, 0].set_title("Exponencial x**5")
axes[1, 0].set_xlabel("Eixo de tempo")
axes[1, 0].set_ylabel("Eixo da amplitude")

axes[0, 1].plot(x, y1)
axes[0, 1].set_title("Exponencial x²")
axes[0, 1].set_xlabel("Eixo de tempo")
axes[0, 1].set_ylabel("Eixo da amplitude")

axes[1, 1].plot(x, y2)
axes[1, 1].set_title("Exponencial x**5")
axes[1, 1].set_xlabel("Eixo de tempo")
axes[1, 1].set_ylabel("Eixo da amplitude")

plt.show()