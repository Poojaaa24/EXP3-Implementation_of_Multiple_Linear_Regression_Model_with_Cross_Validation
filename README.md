# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries & Load Dataset
2. Divide the dataset into training and testing sets.
3. Select a suitable ML model, train it on the training data, and make predictions.
4. Assess model performance using metrics and interpret the results.


## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: POOJA U
RegisterNumber: 212225230209

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt

df=pd.read_csv('CarPrice_Assignment.csv')
#1. LOAD AND PREPARE DATA
data= df.drop(['car_ID', 'CarName'], axis=1)
data = pd.get_dummies(data, drop_first=True)
#2. SPLIT DATA
X=data.drop('price', axis=1)
y=data['price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
#3. CREATE AND TRAIN MODEL
model= LinearRegression()
model.fit(X_train,y_train)
print('Name: POOJA U ')
print('Reg.No: 21222523209 ')
print("\n=== Cross-Validation ===")
cv_scores= cross_val_score(model,X,y, cv=5)
print(f"Fold R2 scores:", ["{score:.4f}" for score in cv_scores])
print(f"Average R2:{cv_scores.mean():.4f}")

#Test set evaluation
y_pred=model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R2: {r2_score(y_test,y_pred):.4f}")
#Visualization
plt.figure(figsize=(8, 6))
plt.scatter(y_test,y_pred, alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted CAR Prices")
plt.show()
 
*/
```

## Output:
<img width="1280" height="661" alt="Screenshot 2026-02-24 230220" src="https://github.com/user-attachments/assets/08355124-a6c2-487b-aadb-cf02f92bc1ff" />
<img width="1256" height="782" alt="Screenshot 2026-02-24 230238" src="https://github.com/user-attachments/assets/6bfe5a20-5e0b-468b-8afd-041b9ea4ae3d" />

<img width="1372" height="752" alt="Screenshot 2026-02-12 103038" src="https://github.com/user-attachments/assets/3ea927f9-8f79-45d4-8fa1-9b9ecacceb6f" />
<img width="1348" height="895" alt="Screenshot 2026-02-12 103055" src="https://github.com/user-attachments/assets/bbda3c28-17b3-40e2-8f90-0f9e399f5983" />




## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
