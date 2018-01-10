import matplotlib.pyplot as plt
import numpy as np

hoi = np.load('repl_mem.npy')

fig1 = plt.figure(1)
plt.imshow(hoi[0][0][0], cmap = 'Greys')
fig1.show()

fig2 = plt.figure(2)
plt.imshow(hoi[0][0][1], cmap = 'Greys')
fig2.show()

input()
