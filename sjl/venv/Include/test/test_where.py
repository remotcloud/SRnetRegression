import numpy as np
a = np.array([1,2,3])
b = np.where(a>1)
print(b[0][1])
print(type(b))
a[b] = 0
print(a)