import numpy as np
import matplotlib.pyplot as plt

# matplotlib - plota gráficos
# numpy - biblioteca matricial


t = np.linspace(0, 2*np.pi, 1000) # Gera array de valores de determinado valor inicial a um valor final

y = np.cos(4*t) # função cosseno | 4*t se diz respeito ao período
y1 = np.sin(4*t) # função seno

plt.figure("Cosseno", figsize=(7, 5)) # Criar gráfico na figura 1 | figsize=(largura, altura)
plt.plot(t, y)
plt.title("Gráfico do Cosseno")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")

plt.figure("Seno", figsize=(7,5)) # Criar gráfico na figura 2
plt.plot(t, y1)
plt.title("Gráfico do Seno")
plt.xlabel("Tempo  (s)")
plt.ylabel("Amplitude")

plt.show()
