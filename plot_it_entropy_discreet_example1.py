"""
===============================
Entropy  -  Discreet Source
===============================


An example of computation of entropy for a discreet source.

"""


import numpy as np
import matplotlib.pyplot as plt
import spkit as sp


# Generate a discreet random variable x

np.random.seed(2)
x = (np.random.rand(30)*10).round()


# compute entropy
Hx = sp.entropy(x,is_discrete=True)

print('Random variable x : \n',x)


x_u,frq = np.unique(x, return_counts=True)
prob = frq/frq.sum()

plt.bar(x_u,prob)
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title(f'Entropy H(x) = {Hx.round(3)} bits')
plt.show()
