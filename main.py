from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
from os import environ

# Load the saved model
with open('./model/heart_disease_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create the FastAPI app
app = FastAPI()  

# Define the input data model
class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Define the prediction endpoint
@app.post("/predict/")
def predict(input_data: HeartDiseaseInput):
    # Convert input data to numpy array
    data = np.array([[
        input_data.age, input_data.sex, input_data.cp, input_data.trestbps,
        input_data.chol, input_data.fbs, input_data.restecg, input_data.thalach,
        input_data.exang, input_data.oldpeak, input_data.slope,
        input_data.ca, input_data.thal
    ]])
    
    # Make prediction
    prediction = model.predict(data)
    probability = model.predict_proba(data).max()

    # Interpret the prediction
    if prediction == 1:
        result = "The person has a heart disease."
    else:
        result = "The person has a healthy heart."

    return {"prediction": int(prediction), "probability": probability, "result": result}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=environ.get("PORT",8000))