# ML DEPLOY - Main module of easy-ml-deploy library
# A library to deploy machine learning models in a simple way using FastAPI 
# Author: @barthemius
# License: MIT

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os


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
        class Input(BaseModel):
            input_data: list

        class Output(BaseModel):
            prediction: list
        
        # Create a new route for the model
        @self.app.post("/" + model_name)
        def predict(input_data: Input):
            # Transform data
            if scaler is not None:
                input_data = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_data)

            # Return prediction
            return JSONResponse(content=jsonable_encoder(Output(prediction=prediction)))
    
    # Function - run the app
    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)





