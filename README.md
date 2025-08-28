# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students
2. Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored
3. Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis
4. Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b
5. for each data point calculate the difference between the actual and predicted marks
6. Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error
7. Once the model parameters are optimized, use the final equation to predict marks for any new input data

## Program:
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M.Mohamed Ashfaq
RegisterNumber: 212224240090
```
/*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df= pd.read_csv('data.csv')

df.head()
df.tail()

X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
y_pred

y_test

import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
*/
```

## Output:
## Head Values:
<img width="316" height="456" alt="headvalues ml exp2" src="https://github.com/user-attachments/assets/585c270c-8c91-472f-b714-12125c5d5138" />

## Tail Values:
<img width="340" height="452" alt="tailvaluesml exp2" src="https://github.com/user-attachments/assets/0fdabda9-1986-4cae-9fa9-fb0897664e24" />

## X Values:
<img width="294" height="1082" alt="xvaluesexp2" src="https://github.com/user-attachments/assets/036f8ce3-2157-414d-97bb-a79640823e14" />


## Y Values:
<img width="1424" height="94" alt="yvaluesexp2" src="https://github.com/user-attachments/assets/2c2846d8-fe63-4783-88ae-a12d4ce2affd" />


## Predicted Values:
<img width="1392" height="152" alt="predicted valuesexp2" src="https://github.com/user-attachments/assets/964695bd-971f-48b2-85b1-5f52ded84a64" />


## Actual Values:
<img width="1142" height="56" alt="actual values exp2" src="https://github.com/user-attachments/assets/5632a3f2-93bd-4caf-ac40-df20632efdf5" />


## Training Set:
<img width="1114" height="890" alt="training set exp2" src="https://github.com/user-attachments/assets/f9534c90-3b15-4941-b03a-d1a211715cb6" />


## Testing Set:
<img width="1104" height="892" alt="training set set exp2" src="https://github.com/user-attachments/assets/da551aa1-c15e-4893-8aeb-263b3b133c70" />


## MSE, MAE and RMSE:
<img width="498" height="120" alt="mse mae rmse exp2" src="https://github.com/user-attachments/assets/869e22f6-98a7-4f6a-95b5-e1e9ac6fb8d8" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
