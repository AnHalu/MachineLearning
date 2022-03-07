import matplotlib.pyplot as plt
import numpy as np

def F(x) :
    return x*x - 5*np.sin(x)

def f(x) :
    return 2*x - 5*np.cos(x)

def GD(LR , x0) :
    """
    Args :
    :param LR: hang so hoc
    :param x0: diem bat dau
    :return: 1 list chua cac diem x
    """
    x = [x0]
    while(1) :
        xms = x[-1] - LR*f(x[-1])

        if abs(f(xms)) < 1e-6 :
            break

        x.append(xms)
    return x

x = GD(0.001, 5)

a = np.linspace(-5, 4, 100)
plt.plot(a, F(a))
plt.show()
plt.savefig("anh.png")
