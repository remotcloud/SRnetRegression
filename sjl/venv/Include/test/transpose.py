import numpy
import numpy as np

data = np.array([[1,2,3,4],
                [4,5,6,7]])
data = data.T
data = list(list(d) for d in data)

#data = np.array(data)
for d in data:
    for a in d:
        print(a)
print(data[0][0])