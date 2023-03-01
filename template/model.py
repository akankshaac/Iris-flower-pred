import pandas as pd
import numpy as np
import sklearn as sklearn
import pickle
"""Algorithm = SVC or LogisticRegression or DecisionTreeClassifier"""

data = pd.read_csv('iris.csv')

x = data.iloc[:,0:4]
y = data.iloc[:,4]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.2)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train.values, y_train.values)
pred = svc.predict()

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(x_train.values, y_train.values)
pred = LR.predict()

from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(x_train.values,y_train.values)
Pred = DTC.predict()

pickle.dump(svc,open('svc_model.pkl','wb'))
pickle.dump(LR,open('LR_model.pkl','wb'))
pickle.dump(DTC,open('DTC_model.pkl','wb'))