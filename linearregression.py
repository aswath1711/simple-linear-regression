#importing packages
import numpy as np                                  #number manipulation
import matplotlib.pyplot as plt                      #plotting graphs
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression   #package having the models
import pandas as pd                                 #data frame
import math                                         #probability operations


#making dataframes 
train_set1=pd.read_csv('train.csv')                 #loading data for training
test_set1=pd.read_csv('test.csv')                   #loading data for test set

#print(train_set1.shape)
#print(pd.DataFrame(train_set1).head(5))
#print(test_set1.shape)
#print(pd.DataFrame(test_set1).head(5))


#loading frames in a variable    dropna()-->null values are dropped
train_set=train_set1.dropna()                       
test_set=test_set1.dropna()


#transforming the dimensions as x and y values
Xtrain=train_set[['x']].as_matrix()
Ytrain=train_set[['y']].as_matrix()
Xtest=test_set[['x']].as_matrix()
Ytest=test_set[['y']].as_matrix()
#print('mean of X is',np.mean(Xtrain),'\n')
#print('median of X is',np.median(Xtrain),'\n')
#print('mean of Y is',np.mean(Ytrain),'\n')
#print('median of Y is',np.median(Ytrain),'\n')



#visualizing the data
plt.title('plot of the given data')
plt.scatter(Xtrain,Ytrain,s=5,c='black',marker='*')
plt.xlabel('training_set_X')
plt.ylabel('training_set_Y')
plt.show()


np_x_train = np.array(Xtrain)
np_y_train = np.array(Ytrain)
np_x_test = np.array(Xtest)
np_y_test = np.array(Ytest)

n = len(np_x_train)
alpha = 0.000001
a = 0
b = 0
for epoch in range(1000):
    y = a * np_x_train + b
    error = y - np_y_train
    mean_sq_er = np.sum(error**2)/n
    b = b - alpha * 2 * np.sum(error)/n 
    a = a - alpha * 2 * np.sum(error * np_x_train)/n
    if(epoch%10 == 0):
        print(mean_sq_er)


np_y_prediction = a * np_x_test + b
print('R2 Score:', r2_score(np_y_test,np_y_prediction))
plt.xkcd()
np_y_plot = []
for i in range(100):
    np_y_plot.append(a * i + b)
plt.figure(figsize=(15,15))
plt.scatter(np_x_test,np_y_test,color='red',label='values')
plt.plot(range(len(np_y_plot)),np_y_plot,color='black',label = 'prediction')
plt.legend()
plt.show()
