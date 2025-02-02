import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
S = (np.exp(x)) / (1 + np.exp(x))
Sprime = S * (1 - S)

plt.plot(x, S, label="sigmoid")
plt.plot(x, Sprime, label="first derivative of sigmoid")
plt.legend()
plt.show()