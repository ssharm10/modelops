import pickle
import joblib
import pandas as pd 
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from fastapi.middleware.cors import CORSMiddleware

# Load the pre-trained model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI(title="Car Price Prediction API", version="1.0.0")

# Add CORS middleware to prevent the browser from blocking requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Create health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Define valid manufacturer-model combinations
VALID_COMBINATIONS = {
    "BMW": ["M5", "X3", "Z4"],
    "Ford": ["Fiesta", "Focus", "Mondeo"],
    "Porsche": ["718 Cayman", "911", "Cayenne"],
    "Toyota": ["Prius", "RAV4", "Yaris"],
    "VW": ["Golf", "Passat", "Polo"]
}

#Create prediction endpoint
@app.post("/predict")
def predict_car_price(payload_dict: dict):
    # Extract features from input dictionary
    manufacturer = str(payload_dict["Manufacturer"]).strip()
    model_name = str(payload_dict["Model"]).strip()
    fuel = str(payload_dict["Fuel type"]).strip()
    engine = float(payload_dict["Engine size"])
    year = int(payload_dict["Year of manufacture"])
    mileage = float(payload_dict["Mileage"])
    
    # Validate manufacturer-model combination
    if manufacturer not in VALID_COMBINATIONS:
        return {"error": f"Invalid manufacturer: {manufacturer}"}
    
    if model_name not in VALID_COMBINATIONS[manufacturer]:
        return {
            "error": f"Invalid combination: {manufacturer} does not make {model_name}. "
                    f"Valid models for {manufacturer} are: {', '.join(VALID_COMBINATIONS[manufacturer])}"
        }
    
    # Derived features (expected by training pipeline)
    CURRENT_YEAR = 2025
    age = max(CURRENT_YEAR - year, 0)
    mileage_per_year = mileage / max(age, 1)
    vintage = int(age >= 20)
    
    # Create single-row dataframe with all features needed by the model
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

#Create Categories endpoint
@app.get("/categories")
def get_categories():
    """
    Get available categories for categorical features only
    """
    try:
        preprocessor = model.named_steps['preprocessor']
        categories_dict = {}
        
        # Iterate through transformers to find categorical features
        for name, transformer, columns in preprocessor.transformers_:
            # Check if transformer has categories (for categorical features)
            if hasattr(transformer, 'categories_'):
                for i, col in enumerate(columns):
                    categories_dict[col] = [str(cat) for cat in transformer.categories_[i]]
            
            # Check if it's a Pipeline with steps
            elif hasattr(transformer, 'named_steps'):
                for step_name, step in transformer.named_steps.items():
                    if hasattr(step, 'categories_'):
                        for i, col in enumerate(columns):
                            categories_dict[col] = [str(cat) for cat in step.categories_[i]]
        
        return {
            "status": "success",
            "categories": categories_dict
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "categories": {}
        }