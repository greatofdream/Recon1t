from numpy.polynomial import legendre as LG
import numpy as np
cut = 10
cos_theta = np.arange(-1,1,0.1)
x = np.zeros((np.size(cos_theta), cut))
for i in np.arange(0,cut):
    c = np.zeros(cut)
    c[i] = 1
    x[:,i] = LG.legval(cos_theta,c)
print(cos_theta)
print(x)
