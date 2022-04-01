import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
data = pd.read_csv('data_Logistic.csv')
x = data.iloc[:, 2:4].values
y = data.iloc[:, 4].values
#preprocessing bang standarScaler
x_pre = preprocessing.StandardScaler().fit_transform(x)
#split
x_train, x_test, y_train, y_test = train_test_split(x_pre[:, 0:2], y, test_size=50)
#LR
LR = LogisticRegression()
LR.fit(x_train, y_train)
#train test
y_pre = LR.predict(x_test)
print('%.2f%%' %(100*accuracy_score(y_test, y_pre)))



