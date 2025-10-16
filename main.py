import pickle
import joblib
import pandas as pd 
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

# Load the pre-trained model
model = joblib.load("../../../../models/model.pkl")

# Create FastAPI app
app = FastAPI(title="Car Price Prediction API", version="1.0.0")

# Create health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Create model metadata endpoint
@app.get("/metadata")
def get_metadata():
    return {
        "model_info": "Car Price Prediction Model",
        "model_type": type(model.named_steps["model"]).__name__,
        #"model_parameters": {k: str(v) for k, v in model.named_steps["model"].get_params().items()},
        "features":  list(model.feature_names_in_)
    }
# Create prediction endpoint
@app.post("/predict")
def predict_car_price(payload_dict: dict):

#Extract features from input dictionary
    manufacturer = str(payload_dict["Manufacturer"]).strip()
    model_name = str(payload_dict["Model"]).strip()
    fuel = str(payload_dict["Fuel type"]).strip()
    engine = float(payload_dict["Engine size"])
    year = int(payload_dict["Year of manufacture"])
    mileage = float(payload_dict["Mileage"])    

# Derived features (expected by training pipeline)
    CURRENT_YEAR = 2025     
    age = max(CURRENT_YEAR - year, 0)
    mileage_per_year = mileage / max(age, 1)
    vintage = int(age >= 20)    

# Create single-row dataframe with all features needed by the model to make the prediction
    row = {
        "Manufacturer": manufacturer,         
        "Model": model_name,
        "Fuel type": fuel,
        "Engine size": engine,
        "Year of manufacture": year,
        "Mileage": mileage,
        "age": age,
        "mileage_per_year": mileage_per_year,
        "vintage": vintage,
    }
    df = pd.DataFrame([row])

    # Make prediction using the pre-trained model
    prediction = model.predict(df)[0]
    return {"predicted_price": float(round(prediction, 2))}

# To run the app locally using fastapi, use the command:
# fastapi dev main.py
# To test the endpoints, you can use:
# 1. Health check: http://localhost:8000/health
# 2. Metadata: http://localhost:8000/metadata
# 3. Prediction: Use a tool like curl or Postman to POST JSON data to http://localhost:8000/predict

#Create root end-point
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_path = os.path.join("templates", "index.html")
    with open(html_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)