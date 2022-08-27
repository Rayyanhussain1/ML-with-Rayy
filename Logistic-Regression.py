from cgi import print_arguments
import pandas as pd
customer=pd.read_csv('customer_churn.csv')
print(customer.head())
x=customer[['tenure']]
y=customer[['Churn']]
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
lg= LogisticRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35)
lg.fit(x_train,y_train)
y_pred=lg.predict(x_test)
print(y_pred)
print(y_test) 
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
print((1750+133)/(1750+76+507+133))