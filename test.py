# Testing script for mldeploy module

# Train simple linear regression model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import mldeploy

# Load data
boston = load_boston()
X = boston.data
y = boston.target

print(X.shape, y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Test model
y_pred = lr.predict(X_test)

# Compute metrics
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

# Train SVM model
svm = SVR(kernel='rbf', C=10, gamma=0.05, epsilon=.1)
svm.fit(X_train, y_train)

# Test model
y_pred = svm.predict(X_test)
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))


# Deploy model
deploy = mldeploy.MLDeploy()
deploy.add_model("linear_regression", lr, scaler)
deploy.add_model("svm", svm, scaler)
deploy.run()
