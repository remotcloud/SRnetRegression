import numpy as np
import matplotlib.pyplot as plt
import sympy
x1 = np.arange(1,100,0.1)
x2 = np.array([0.1748782352400335,0.9235431860779176])
x3 = np.array([1.5407546619558103,1.4597380831819320])
z = (x1+12)/(x1+10)
plt.figure()
plt.plot(x1,z)
plt.show()
x,y =sympy.symbols('x y')
expr = (x+10)/(x+15)
expr2 = 1-5/(x+15)
expr3 = expr - expr2
print(expr3)
print(expr - expr2)
print(expr)