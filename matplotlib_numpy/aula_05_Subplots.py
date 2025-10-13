import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize

x = np.arange(0, 5, 0.1) # 0, 0.1, 0.2, ..., 4.9
y1 = x**2
y2 = x**5
# subplots
fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(nrows = 2, ncols = 2, figsize=(8, 4))
plt.suptitle("Gráficos com Subplots")
plt.subplots_adjust(
    left = 0.093,
    right = 0.948,
    top = 0.8,
    bottom = 0.148,
    wspace = 0.384,
    hspace = 0.824
)

ax1.plot(x, y1)
ax1.set_title("Exponencial x²")
ax1.set_xlabel("Eixo de tempo")
ax1.set_ylabel("Eixo da amplitude")



ax2.plot(x, y2)
ax2.set_title("Exponencial x**5")
ax2.set_xlabel("Eixo de tempo")
ax2.set_ylabel("Eixo da amplitude")

ax3.plot(x, y1)
ax3.set_title("Exponencial x²")
ax3.set_xlabel("Eixo de tempo")
ax3.set_ylabel("Eixo da amplitude")

ax4.plot(x, y2)
ax4.set_title("Exponencial x**5")
ax4.set_xlabel("Eixo de tempo")
ax4.set_ylabel("Eixo da amplitude")

plt.show()



