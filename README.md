# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Start the program.
2. Import the required libraries.
3. Load the California Housing dataset.
4. Select the first three input features.
5. Create two output targets.
6. Split the dataset into training and testing sets.
7. Apply feature scaling using StandardScaler.
8. Initialize the SGDRegressor model.
9. Wrap the model using MultiOutputRegressor.
10. Train the model using the scaled training data.
11. Predict the output values.
12. Convert the predicted values back to their original scale.
13. Calculate the Mean Squared Error (MSE).
14. Compare actual and predicted values.
15. Stop the program.

## Program:
```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Krisha Princy N S
RegisterNumber: 212224060126


# 1. Import libraries
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# 2. Load dataset
data = fetch_california_housing()

# Select first 3 features
X = data.data[:, :3]

# Create two output targets
Y = np.column_stack((data.target, data.data[:, 6]))

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Example X:", X[0])
print("Example Y:", Y[0])

# 3. Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("Train shapes:", X_train.shape, Y_train.shape)
print("Test shapes:", X_test.shape, Y_test.shape)

# 4. Feature scaling
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

print("Scaled X_train mean:", X_train_scaled.mean(axis=0))
print("Scaled Y_train mean:", Y_train_scaled.mean(axis=0))

# 5. Initialize and train model
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
multi_output_sgd = MultiOutputRegressor(sgd)

multi_output_sgd.fit(X_train_scaled, Y_train_scaled)

# 6. Predict and inverse transform
Y_pred_scaled = multi_output_sgd.predict(X_test_scaled)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
Y_test_orig = scaler_Y.inverse_transform(Y_test_scaled)

print("First 5 predictions:")
print(Y_pred[:5])

# 7. Evaluate model performance
mse = mean_squared_error(Y_test_orig, Y_pred)
print("Mean Squared Error:", mse)

mse_per_output = np.mean(
    (Y_test_orig - Y_pred) ** 2, axis=0
)
print("MSE per output:", mse_per_output)

# 8. Compare actual and predicted values
for i in range(5):
    print(f"Example {i+1}")
    print("Inputs:", X_test[i])
    print("True outputs:", Y_test_orig[i])
    print("Predicted:", Y_pred[i])
    print("-" * 40)

```

## Output:

<img width="813" height="861" alt="image" src="https://github.com/user-attachments/assets/1d2ef6fa-6783-469d-b84a-b115456f6fbf" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
