# ML DEPLOY - Main module of easy-ml-deploy library
# A library to deploy machine learning models in a simple way using FastAPI 
# Author: @barthemius
# License: MIT

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import numpy as np


# Class - wrapper of FastAPI app
class MLDeploy:
    def __init__(self, port=8000, host="127.0.0.1"):
        self.app = FastAPI()
        self.port = port
        self.host = host
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Function - add a model to the app
    # model - a tensorflow or sklearn model with a predict method
    # scaler - a sklearn scaler with a transform method
    def add_model(self, model_name, model=None, scaler=None):
        # TODO: Add support for many model API - each model should be in a separate app and have its own route
        # Then, the main app should have a route to each model

        # Check if model is not None
        if model is None:
            raise Exception("Model is None")

        # Check if model has a predict method
        if not hasattr(model, "predict"):
            raise Exception("Model has no predict method")
        
        # Check if scaler has a transform method
        if scaler is not None and not hasattr(scaler, "transform"):
            raise Exception("Scaler has no transform method")
        
        # If everything is ok create a new class for input and output
        class InputModel(BaseModel):
            input_data: List[float]

        class OutputModel(BaseModel):
            prediction: list
        
        # Create a new route for the model
        @self.app.post("/" + model_name)
        def predict(data: InputModel):
            # Validate input data
            if len(data.input_data) == 0:
                return JSONResponse(status_code=400, content={"error": "Input data is empty"})
            
            # Transofrm to numpy array
            input_data = np.array(data.input_data)

            # Check the shape of the input data           
            input_data = input_data.reshape(1, -1)

            # Transform data
            if scaler is not None:
                input_data = scaler.transform(input_data)

            # Predict
            prediction = list(model.predict(input_data))

            # Return prediction
            return JSONResponse(content=jsonable_encoder(OutputModel(prediction=prediction)))
    
    # Function - run the app
    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)





