import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pd.set_option('display.width', 320)
pd.set_option('display.max_columns',10)
data = pd.read_csv('taxi.csv')

# print(data.head())
# print(data.columns)

data_x = data.iloc[:,0:-1].values
data_y = data.iloc[:,-1].values                    #---target or feature column
# print(data_y)
# print(data_x)

X_train,X_test,y_train,y_test =  train_test_split(data_x,data_y,test_size=.3, random_state=0)  # matching value of training and tseting

reg = LinearRegression()
reg.fit(X_train,y_train)    #---Provide training data to reg.fit method

print("Training score",reg.score(X_train,y_train))      # ---Training accuracy
print("Test score",reg.score(X_test,y_test))      # ---Testing accuracy

#-----------creation of model

pickle.dump(reg, open('taxi.pkl','wb'))


#------------testing of model

model = pickle.load(open('taxi.pkl','rb'))
print(model.predict([[80,1770000,6000,85]]))
