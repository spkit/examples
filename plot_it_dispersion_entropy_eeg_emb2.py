"""
===============================
Dispersion Entropy with Embedding dim
===============================

Dispersion Entropy is computed by first discretising the signal and then extracting all the
dispersing patterns of given embedding dimension. The ditrubution of patterns determines the
dispersion entropy of signal. If signal have a few patterns with high repetitions compare to others
signal is less random which entails the low entropy. On the other hand, a random signal with no
patterns repetitions more than others leads to high entropy.

"""

import numpy as np
import matplotlib.pyplot as plt
import spkit as sp


# load sample EEG Signal

X,ch_names = sp.load_data.eegSample()
fs=128
Xi = sp.filter_X(X[:,0],band=[1,20],btype='bandpass',verbose=0)
print('EEG Sample X: Shape',X.shape)


t = np.arange(Xi.shape[0])/fs
plt.figure(figsize=(15,3))
plt.plot(t,Xi)
plt.xlim([0,t[-1]])
plt.xlabel('time (s)')
plt.grid()
plt.show()


#Dispersion Entropy
print('-'*50)
print('Dispersion Entropy with embeding dimention=2')
print('-'*50)



de,prob,patterns_dict,_,_= sp.dispersion_entropy(Xi,classes=10, scale=1, emb_dim=2, delay=1,return_all=True)
print('Disperssion Entropy: ',de)

plt.figure(figsize=(5,4))
plt.stem(np.arange(len(prob)),prob)
plt.xlabel('pattern #')
plt.ylabel('probability')
plt.title(f'Disperssion Entropy: {de}')
plt.show()


print('All the patterns found (embeding dimention=2)')
sp.utils.pretty_print(patterns_dict,n=5,show_index=False)
