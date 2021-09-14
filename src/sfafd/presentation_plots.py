import numpy as np
import matplotlib.pyplot as plt

from . import sfa
tlen = 2000

t = np.linspace(0, 500, num=tlen)
SF1 = np.cos(np.pi/60 * t) + t/250
SF2 = np.sin(np.pi/10 * t)

X = np.zeros((tlen, 2))
X[:, 0] = 1.4 * SF1 + 2.5 * SF2
X[:, 1] = 1.2 * SF1 - SF2

trainer = sfa.SFA(data=X.T,
                  dynamic_copies=0,
                  expansion_order=1)
Y = trainer.train()

plt.subplot(2, 2, 1)
plt.title("Latent Functions")
plt.plot(SF1, label=r'$f(t)$')
plt.plot(SF2, label=r'$g(t)$')
plt.legend()
plt.xlabel('Time')

plt.subplot(2, 2, 2)
plt.title("Observed Data")
plt.plot(X[:, 0], label=r'$X_1(t)$')
plt.plot(X[:, 1], label=r'$X_2(t)$')
plt.legend()
plt.xlabel('Time')

plt.subplot(2, 2, 3)
plt.title("Recovered Latent Data")
plt.plot(Y.T[:, 0], 'g', label=r'$Y_1(t)$')
plt.plot(Y.T[:, 1], 'r', label=r'$Y_2(t)$')
plt.legend()
plt.xlabel('Time')

plt.subplot(2, 2, 4)
plt.title("Slow Feature Comparison")
plt.plot(SF1, label='Original')
plt.plot(Y.T[:, 0], label='Recovered')
plt.legend()
plt.xlabel('Time')

plt.show()
