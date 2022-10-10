# Easy-ml-deploy

## Overview
Deployment of machine learning models is an essential part of their lifecycle.
Usually the general way for it is to create a RESTful webapp, which methods allow us to use predictive powers of our models.

However, some data scientists struggle with creating such apps from scratch. And here comes easy-ml-deploy with rescue.

## Usage
Easy-ml-deploy allows us to deploy ML models (so far scikit-learn and tensorflow) models with only several lines of code.

```python
from mldeploy import MLDeploy

deploy = MLDeploy()
deploy.add_model("linear_regression", lr, scaler)
deploy.add_model("svm", svm, scaler)
deploy.run()
```