import pandas as pd
                     #DECISION-TREE-CLASSIFIER

iris=pd.read_csv('iris.csv')
print(iris.head())
y=iris[['Species']]
x=iris[['Sepal.Length']]
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.45)
dtc=DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred=dtc.predict(x_test)
print(y_pred)
print(y_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
print((21+14+10)/(21+1+1+2+14+10+0+4+15))

                     #DECISION-TREE-REGRESSOR

iris=pd.read_csv('iris.csv')
print(iris.head())
y=iris[['Sepal.Width']]
x=iris[['Sepal.Length']]
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.45)
dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)
y_pred=dtr.predict(x_test)
print(y_pred)
print(y_test)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))