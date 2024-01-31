import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Score.csv')
# take all the rows and all the columns except the last one
X = dataset.iloc[:, :-1].values
# take all the rows and the last column
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# 80% of the data is used for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
# use the LinearRegression class to create a regressor object
regressor = LinearRegression()
# train the model using the training sets
regressor.fit(X_train, y_train)

# Predicting the Training set results
y_pred = regressor.predict(X_test)


# Visualising the Training and Test set results simultaneously
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot for Training set
axes[0].scatter(X_train, y_train, color='red')
axes[0].plot(X_train, regressor.predict(X_train), color='blue')
axes[0].set_title('GPA vs SAT Score (Training set)')
axes[0].set_xlabel('SAT Score')
axes[0].set_ylabel('GPA')

# Plot for Test set
axes[1].scatter(X_test, y_test, color='red')
axes[1].plot(X_train, regressor.predict(X_train), color='blue')  # Note: Using X_train here for consistency in plot
axes[1].set_title('GPA vs SAT Score (Test set)')
axes[1].set_xlabel('SAT Score')
axes[1].set_ylabel('GPA')

plt.tight_layout()
plt.show()


# Making a single prediction (for example the GPA of a student with SAT score 1600)
print(regressor.predict([[1600]]))

# Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)

