import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv('data_linear.csv')
reg = linear_model.LinearRegression(fit_intercept= False)

Xbar = np.concatenate((np.ones((30, 1)), data.iloc[:, 0:1]), axis=1)
Y = data.iloc[:,1]

reg.fit(Xbar, Y)
# y = w0 + x1*w1
w = reg.coef_

ax = np.linspace(30,100)


plt.plot(ax, (w[0] + ax*w[1]), 'r-')
plt.plot(Xbar[:, 1], Y, 'bx')

plt.show()


