from sklearn import datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

X0 = iris_x[iris_y == 0, 0:2]
X1 = iris_x[iris_y == 1, 0:2]
X2 = iris_x[iris_y == 2, 0:2]
X = np.concatenate((X0, X1), axis=0)

Y = np.concatenate((np.ones((1, X0.shape[0])), -1*np.ones((1, X1.shape[0]))), axis=1)
Y = Y.reshape(100)
clf = SVC(kernel='linear', C=1e5)
clf.fit(X, Y)

w = clf.coef_
b = clf.intercept_

ax = np.linspace(min(X[:, 0]), max(X[:, 0]))

plt.plot(X0[:, 0], X0[:, 1], 'rx')
plt.plot(X1[:, 0], X1[:, 1], 'bo')

w1 = w[0][0]
w2 = w[0][1]

plt.plot(ax, -(b+ax*w1)/w2, 'y')
plt.plot(ax, -(b+ax*w1+1)/w2, 'y--')
plt.plot(ax, -(b+ax*w1-1)/w2, 'y--')

plt.show()

"""
chon 2 cot du lieu dau tien ( 0 , 1 ) cua data iris : chieu dai va rong cua dai hoa de xet 
chon 2 class 0 và 1 ( 2 trong 3 loai hoa ) 
dua tren chieu dai va rong de tach ra thanh 2 loai do  
xu ly duoc bai toan classification phan loai ra 2 loai hoa 
"""