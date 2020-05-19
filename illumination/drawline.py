import numpy as np 
import matplotlib.pyplot as plt 

x = np.linspace(0, 1, 300)

y1 = [pow(i, 0.1) for i in x]
y2 = [pow(i, 0.3) for i in x]
y3 = [pow(i, 0.5) for i in x]
y4 = [pow(i, 0.7) for i in x]
# y5 = [pow(i, 0.9) for i in x]

y6 = [pow(i, 1) for i in x]

y7 = [pow(i, 1.5) for i in x]
y8 = [pow(i, 2.5) for i in x]
y9 = [pow(i, 3.5) for i in x]
y10 = [pow(i, 4.5) for i in x]


plt.plot(x, y1)
plt.text(0.03, 0.77, "a = 0.1")
plt.plot(x, y2)
plt.text(0.10, 0.56, "a = 0.3")
plt.plot(x, y3)
plt.text(0.095, 0.34, "a = 0.5")
plt.plot(x, y4)
plt.text(0.10, 0.23, "a = 0.7")
# plt.plot(x, y5)
plt.plot(x, y6)
plt.text(0.12, 0.16, "a = 1")
plt.plot(x, y7)
plt.text(0.35, 0.24, "a = 1.5")
plt.plot(x, y8)
plt.text(0.41, 0.16, "a = 2.5")
plt.plot(x, y9)
plt.text(0.46, 0.09, "a = 3.5")
plt.plot(x, y10)
plt.text(0.65, 0.18, "a = 4.5")
plt.show()
