
# coding: utf-8

# In[1]:

import numpy as np
import heapq as hq
np.random.seed(0)


# In[2]:

T = 8
N = 2
maxReg = np.array([0, 0])
lastReg = np.array([0, 0])
M = np.zeros(T)
L = np.zeros(T)
for ii in np.arange(0, T):
    num = np.random.randint(1, 10, 1)
    if lastReg[-2] == 0:
        lastReg[-1] = num
    lastReg[-2] = lastReg[-1]
    lastReg[-1] = num
    if num > np.min(maxReg):
        maxReg[maxReg == np.min(maxReg)] = num
    M[ii] = np.prod(maxReg)
    L[ii] = np.prod(lastReg)
diffML = M - L
print("Mean of M - L: ", np.mean(diffML))
print("Standard deviation of M - L: ", np.std(diffML))
a = 32
b = 64
p_diffML_in_ab = len(diffML[(diffML>=a) * (diffML<=b)])/len(diffML)
p_diffML_leq_b = len(diffML[diffML<=b])/len(diffML)
print("Probability that M - L >= a given M - L <= b: ", p_diffML_in_ab/p_diffML_leq_b)


# In[3]:

T = 32
N = 4
maxReg = np.zeros(N)
lastReg = np.zeros(N)
M = np.zeros(T)
L = np.zeros(T)
for ii in np.arange(0, T):
    num = np.random.randint(1, 10, 1)
    if lastReg[0] == 0:
        lastReg[:] = num
    lastReg[-2] = lastReg[-1]
    lastReg[-1] = num
    if num > np.min(maxReg):
        maxReg[maxReg == np.min(maxReg)] = num
    M[ii] = np.prod(maxReg)
    L[ii] = np.prod(lastReg)
diffML = M - L
print("Mean of M - L: ", np.mean(diffML))
print("Standard deviation of M - L: ", np.std(diffML))
a = 2048
b = 4096
p_diffML_in_ab = len(diffML[(diffML>=a) * (diffML<=b)])/len(diffML)
p_diffML_leq_b = len(diffML[diffML<=b])/len(diffML)
print("Probability that M - L >= a given M - L <= b: ", p_diffML_in_ab/p_diffML_leq_b)


# In[ ]:



