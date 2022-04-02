import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('Mall.csv')
k = 4
x = data.iloc[:,2:4].values
kmeans = KMeans(n_clusters=k)
kmeans.fit(x)
labels = kmeans.fit_predict(x)
color = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(k):
    plt.scatter(x[labels == i,0], x[labels == i,1], c=color[i], label=('Cluster' + str(i)))
plt.show()