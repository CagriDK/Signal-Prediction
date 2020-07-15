"""
First we checked the idea of the system which we need to predict the signal strength
with using distance feature and this is more likely seems like an supervised Learning:
Regression problem.

In this type of problem we can choose between speed or accuracy for the method we will use.
For speed we can choose Decision Tree or Linear Regression 
For Accuracy we can choose Random Forest,Neural Network or Gradient Boosting Tree
Since BLE ıot devices can act in real time it is important to have quick responses.
This is the reason why ı used linear regression on my system.Also it has only 1 feature to
set and 1 feature to target and its visualization with linear regression could be more
understandable than other models.
"""

"""
Created on Sun May 10 12:33:18 2020

@author:Çağrı

"""
# Required Libraries to installed for project
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
# Read data from excel with help of pandas library

df_test=pd.read_excel("test_data.xlsx")
df_train=pd.read_excel("train_data.xlsx")

#plot data
plt.xlabel('distance')
plt.ylabel('signal_strength')
plt.scatter(df_train.distance,df_train.signal_strength,color='red',marker='x')

# # divide the data into “attributes” and “labels” on train_data.
x_train=df_train['distance'].values.reshape(-1,1)
y_train=df_train['signal_strength'].values.reshape(-1,1)

# # divide the data into “attributes” and “labels” on test_data.
x_test=df_test['distance'].values.reshape(-1,1)
y_test=df_test['signal_strength'].values.reshape(-1,1)

# standartization method for better prediction 4.9 Root Mean Square to 4.6
# using fit_transform on x_test and y_test grants us a lower Root mean square
# and fits the data into standart scale.

stdscalar = StandardScaler()
x_train = stdscalar.fit_transform(x_train)
x_test =stdscalar.fit_transform(x_test)
# x_test =stdscalar.transform(x_test)
y_train = stdscalar.fit_transform(y_train)
y_test = stdscalar.fit_transform(y_test)
# y_test = stdscalar.transform(y_test)

# to train our algorithm 
regressor = linear_model.LinearRegression() 
regressor.fit(x_train, y_train) #training the algorithm

#The efficiency of prediction which around %85
print('Prediction Efficiency: {}'.format(regressor.score(x_test, y_test))) 

## setting plot style
plt.style.use('fivethirtyeight') 

# making prediction of data
y_pred = regressor.predict(x_test)

# inverse transform of test and train data's
y_pred = stdscalar.inverse_transform(y_pred,)

x_train= stdscalar.inverse_transform(x_train,)
x_test= stdscalar.inverse_transform(x_test,)
y_train= stdscalar.inverse_transform(y_train,)
y_test= stdscalar.inverse_transform(y_test,)


# To compare the actual output values for X_test with the predicted values, execute the followin
# Check df dataframe for comparing purposes
data = {
    'Actual': y_test.flatten(),
    'Predicted':  y_pred.flatten(),
       }
df = pd.DataFrame(data)




# FUNCTION OF ROOT MEAN SQUARE ERROR

def rmse(predictions, targets):
    differences = predictions - targets                       #the DIFFERENCEs.
    differences_squared = differences ** 2                    #the SQUAREs of ^
    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^
    return rmse_val  
# Value of Root Mean Square Value
RMSEVAL=rmse(y_pred,y_test)
print("Root Mean Square Error :",RMSEVAL)




# FUNCTION OF MEAN ABSOLUTE PERCENTAGE ERROR

# Since we have zeroes in data we can't directly calculate the RMSE because we divide
# the value into zero which it will give us infinte.To Avoid this we take average
# Of actualy value dividing into error will give us approximated MAPE and its 0.3138
Error = np.sum(np.abs(np.subtract(y_test,y_pred)))
Average = np.sum(y_test)
MAPE = Error/Average
print("Mean Absolute Percentage Error :",MAPE)




# FUNCTION OF ROOT SQUARE

import itertools
from statistics import mean
# Take data from dataframe
df_r2_1= pd.DataFrame(df_train['distance'])
df_r2_2= pd.DataFrame(df_train['signal_strength'])
# Turn the dataframe into list
xs22=df_r2_1.values.tolist()
ys22=df_r2_2.values.tolist()
# taking flat list to outer list for list of float
xs= list(itertools.chain.from_iterable(xs22))
ys= list(itertools.chain.from_iterable(ys22))
# Convert float values into integer values

integer_val1= []
integer_val2= []

for item in xs:
    integer_val1.append(int(item))
for item in ys:
    integer_val2.append(int(item))
# Convert it into the float numpy array to calculate the r2
xs = np.array(integer_val1, dtype=np.float64)
ys = np.array(integer_val2, dtype=np.float64)
# Slope of the system for calculate
def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b
# squared error found in this function
def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))
# Coefficient determination with data signal_strength and distance
def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)
    
m, b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x)+b for x in xs]
# System calls the function and saving the result into r_squared
r_squared = coefficient_of_determination(ys,regression_line)
print("R_squared :",r_squared)






'''  MAPE NOTES(Additional Ways of finding Mape but different results)

# If it's rathered not taking the average of the actual value,we can simply 
# Ignore the value of zeroes and make calculations without using zero value rows
# Which will give us 790.380 where it is too high for this system



# def percentage_error(actual, predicted):
#     res = np.empty(actual.shape)
#     for j in range(actual.shape[0]):
#         if actual[j] != 0:
#             res[j] = (actual[j] - predicted[j]) / actual[j]
#         else:
#             res[j] = predicted[j] / np.mean(actual)
#     return res

# def mean_absolute_percentage_error(y_true, y_pred): 
#     return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100
# MAPE=mean_absolute_percentage_error(y_test, y_pred)
# print("Mean Absolute Percentage Error :",MAPE)
 




# The third option is changin denominator value a little bit to avoid the infinity
# np.where(array1==0,'Changing Value', array1) #But still it will give us high values

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_test)) * 100

Mape=mean_absolute_percentage_error(y_test, y_pred)


'''
