import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import neighbors
from sklearn.metrics import accuracy_score

data = pd.read_csv('Network_Ads.csv')

impute = SimpleImputer(missing_values=np.nan, strategy='mean')
data.iloc[:, 1:3] = impute.fit_transform(data.iloc[:, 1:3])

x_test = data.iloc[:50, 1:3]
y_test = data.iloc[:50, 3]

x_train = data.iloc[50:, 1:3]
y_train = data.iloc[50:, 3]

res = neighbors.KNeighborsClassifier(n_neighbors=10, p=2, weights='distance')
res.fit(x_train, y_train)
y_pre = res.predict(x_test)

print('%.2f%%' %(100*accuracy_score(y_test, y_pre)))



'''
chon 3 cột age , estimatedSalary , purchased : 
dữ liệu train là cột age , estimatedsalary  
cột đầu ra là purchased 
dự đoạn đầu ra xem có giộng với ytest ko 
'''