'''
Author          : Loo Hui Kie
Contributors    : -
Title           : Dart_Distance_Prediction
Date Released   : 4/1/2024
'''
# Import Necessary Library
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the datasets
train_data_path = r'train.csv'  # Make sure the CSV file exists in the correct directory
validation_data_path = r'validation.csv'
test_data_path = r'test.csv'

train_data = pd.read_csv(train_data_path)
validation_data = pd.read_csv(validation_data_path)
test_data = pd.read_csv(test_data_path)

# Data Exploration and Visualization
print(train_data.describe())
print(train_data[['Velocity (m/s)', 'Distance (m)']].head())

# Plot velocity vs. distance for training data
plt.scatter(train_data['Velocity (m/s)'], train_data['Distance (m)'], color='blue')
plt.xlabel("Velocity (m/s)")
plt.ylabel("Distance (m)")
plt.title("Velocity vs Distance")
plt.show()

# Train-Test Split (80-20 split)
train = train_data[['Velocity (m/s)', 'Distance (m)']]
test = test_data[['Velocity (m/s)', 'Distance (m)']]
x_train = train[['Velocity (m/s)']].values
y_train = train['Distance (m)'].values
x_test = test[['Velocity (m/s)']].values
y_test = test['Distance (m)'].values

# Linear Regression Model
model = LinearRegression()
model.fit(x_train, y_train)

# The coefficients and intercept
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Plot the training data and the regression line
plt.scatter(x_train, y_train, color='blue')
plt.plot(x_train, model.coef_[0] * x_train + model.intercept_, color='red')
plt.xlabel("Velocity (m/s)")
plt.ylabel("Distance (m)")
plt.title("Linear Regression Line - Training Data")
plt.show()

# Predictions and Evaluation on Test Data
y_test_pred = model.predict(x_test)
print("Test MSE: %.2f" % mean_squared_error(y_test, y_test_pred))
print("Test MAE: %.2f" % mean_absolute_error(y_test, y_test_pred))
print("R2-score: %.2f" % r2_score(y_test, y_test_pred))

# Plot Actual vs Predicted values
plt.scatter(x_test, y_test, color='blue', label='Actual')
plt.plot(x_test, y_test_pred, color='red', label='Predicted')
plt.xlabel("Velocity (m/s)")
plt.ylabel("Distance (m)")
plt.legend()
plt.title("Actual vs Predicted Distance")
plt.show()

# Save the model for future use
joblib.dump(model, 'dart_distance_model.pkl')
