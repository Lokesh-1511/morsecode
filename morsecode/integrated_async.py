# File: integrated_async.py
# Purpose: The final, unified system designed to make a REAL, SYNCHRONOUS call to the Gemini API.

from flask import Flask, request, jsonify
import os
import sys
import datetime
import json
import requests
import base64
import time
import warnings
from types import SimpleNamespace

# Heavy data/ML libraries are imported lazily inside functions so the module
# can be imported for smoke tests without installing large wheels (pandas, numpy, etc.).

# --- Configuration (REAL API CALL) ---
# NOTE: Read Gemini API Key from environment to avoid accidental secrets in source.
API_KEY = os.environ.get('GEMINI_API_KEY', '')
MODEL_FILE = 'scalable_price_predictor_pipeline_v3.pkl' 

# FAST_MODE: set environment variable FAST_MODE=1 to train a much smaller mock dataset for smoke tests.
FAST_MODE = bool(os.environ.get('FAST_MODE'))
SAMPLE_SIZE = int(os.environ.get('MOCK_SAMPLES', '2000')) if FAST_MODE else 100000

# Ignore unnecessary warnings during training/loading
warnings.filterwarnings('ignore') 

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# --- Gemini API Configuration ---
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"

# System instructions and Schema (Standard)
SYSTEM_PROMPT = """
You are an expert agricultural produce grader. Your task is to analyze the provided image of a crop and assign it one of three quality grades: 'A', 'B', or 'C'.
"""
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "quality_grade": {"type": "STRING", "enum": ["A", "B", "C", "X"]},
        "justification": {"type": "STRING"}
    },
    "required": ["quality_grade", "justification"]
}

# --- Model Integration Mappings ---
GRADE_TO_SCORE_MAP = {
    'A': 9.5, 'B': 7.5, 'C': 5.5   
}

# --- Mock Data Mappings ---
PRODUCT_NAME_TO_CODE_MAP = {
    "Tomato (Local)": "Product_01", "Basmati Rice": "Product_02", "Chili (Dried)": "Product_03", "Onion (Nasik)": "Product_04",
    "Cotton (Long Staple)": "Product_05", "Ginger (Fresh)": "Product_06", "Moong Dal": "Product_07", "Potatoes (Red)": "Product_08",
    "Mustard Oil Seed": "Product_09", "Millet (Jowar)": "Product_10",
}
def generate_region_map(n=500):
    states = ["Maharashtra", "Odisha", "Gujarat", "Karnataka", "Punjab", "Rajasthan", "Telangana", "Delhi-NCR"]
    region_map = {}
    for i in range(n):
        state = np.random.choice(states)
        code = f"Region-{i:03d}_{state}"
        friendly_name = f"Market_{np.random.randint(10, 999)}-{state.split('-')[0][:3]}"
        region_map[friendly_name] = code
    region_map["Cuttack Market"] = "Region-385_Odisha"
    return region_map
# Create a small static region map for FAST_MODE to avoid importing numpy at import time.
if FAST_MODE:
    REGION_NAME_TO_CODE_MAP = {
        "Cuttack Market": "Region-385_Odisha",
        "Default Market": "Region-000_Default"
    }
else:
    # import numpy lazily for full mode
    import numpy as np
    REGION_NAME_TO_CODE_MAP = generate_region_map()

# --- Model Training/Loading Functions (DEFINED FIRST) ---

def create_scalable_mock_data(n_samples=None):
    """Generates mock data. For FAST_MODE (smoke tests) we use fewer samples."""
    if n_samples is None:
        n_samples = SAMPLE_SIZE
    print(f"Generating mock data and training a new model. (n_samples={n_samples})")
    # Import heavy libs locally
    import numpy as np
    import pandas as pd

    product_codes = list(PRODUCT_NAME_TO_CODE_MAP.values())
    region_codes = list(REGION_NAME_TO_CODE_MAP.values())
    mock_grades = np.random.choice(list(GRADE_TO_SCORE_MAP.keys()), n_samples, p=[0.4, 0.4, 0.2])
    base_quality_scores = np.array([GRADE_TO_SCORE_MAP[g] + np.random.normal(0, 0.5) for g in mock_grades])
    base_quality_scores = np.clip(base_quality_scores, 5.0, 10.0)

    df = pd.DataFrame({
        'Product_Name': np.random.choice(product_codes, n_samples), 'Harvest_Month': np.random.randint(1, 13, n_samples),
        'Region_Code': np.random.choice(region_codes, n_samples), 'Quality_Score': base_quality_scores, 
        'Pest_Damage_Ratio': np.clip(np.random.lognormal(-2.5, 1.0, n_samples), 0.0, 0.1),
        'Market_Demand_Index': np.clip(np.random.normal(1.2, 0.4, n_samples), 0.5, 2.5),
        'Storage_Cost_Index': np.clip(np.random.normal(0.2, 0.1, n_samples), 0.05, 0.5),
        'Current_Temperature_C': np.clip(np.random.normal(30, 8, n_samples), 10, 45),
        'Last_7_Days_Rainfall_MM': np.clip(np.random.lognormal(2.5, 1.0, n_samples), 0.0, 300),
    })
    
    price_noise = np.random.normal(0, 5, n_samples)
    price_by_product = df['Product_Name'].apply(lambda x: {'Product_01': 20, 'Product_02': 60, 'Product_09': 70}.get(x, 45))
    df['Base_Price_INR'] = (
        price_by_product + df['Quality_Score'] * 3  - df['Pest_Damage_Ratio'] * 50 
        + df['Market_Demand_Index'] * 10 + df['Storage_Cost_Index'] * 20 
        - df['Current_Temperature_C'] * 0.1 - df['Last_7_Days_Rainfall_MM'] * 0.05 + price_noise
    ).round(2)
    df['Base_Price_INR'] = np.clip(df['Base_Price_INR'], 10.0, 150.0)
    return df

def train_and_save_model(df):
    # Import heavy libs locally
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from category_encoders import TargetEncoder
    import lightgbm as lgb
    import joblib

    X = df.drop('Base_Price_INR', axis=1)
    y = df['Base_Price_INR']
    categorical_features = ['Product_Name', 'Harvest_Month', 'Region_Code']
    numerical_features = ['Quality_Score', 'Pest_Damage_Ratio', 'Market_Demand_Index', 'Storage_Cost_Index', 'Current_Temperature_C', 'Last_7_Days_Rainfall_MM']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', TargetEncoder(min_samples_leaf=20, smoothing=10.0), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ], remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lgb.LGBMRegressor(random_state=42, n_estimators=500, learning_rate=0.05, n_jobs=-1))
    ])
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Training complete. Pipeline saved to {MODEL_FILE}")
    return pipeline

def load_or_train_model():
    # In FAST_MODE provide a lightweight dummy predictor to avoid heavy imports
    if FAST_MODE:
        print("FAST_MODE enabled: using lightweight dummy predictor.")
        # Dummy pipeline with a predict method
        class DummyPredictor:
            def predict(self, X):
                # Return average price based on a simple heuristic
                return [45.0]
        return DummyPredictor()

    # Full loading/training path (heavy libs imported inside train_and_save_model)
    import joblib
    if os.path.exists(MODEL_FILE):
        print(f"Loading existing model from {MODEL_FILE}...")
        try:
            pipeline = joblib.load(MODEL_FILE)
            if len(pipeline.feature_names_in_) != 9: raise ValueError("Feature count mismatch. Forcing retraining.")
            print("Model loaded successfully. API is ready.")
            return pipeline
        except Exception:
            print("Model load failed or feature structure mismatch. Forcing retraining...")
            df = create_scalable_mock_data()
            pipeline = train_and_save_model(df)
            return pipeline
    else:
        df = create_scalable_mock_data()
        pipeline = train_and_save_model(df)
        return pipeline

# --- Core Image Grading Logic (Integrated and Simulated) ---

def internal_call_gemini_api(base64_image_data):
    """
    Attempts the real call. If API_KEY is missing or the network fails, it uses a simulation fallback.
    """
    
    if not API_KEY or API_KEY == "":
        # 🟢 SIMULATION FALLBACK: Guarantees stable Grade B for the demo if key is missing/invalid.
        return {
            "quality_grade": "B", 
            "justification": "Grade B SIMULATED for demonstration stability (API key not configured/network blocked)."
        }
    
    # --- REAL EXTERNAL API CALL LOGIC ---
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Analyze the produce in this image and assign a quality grade."}, {"inlineData": {"mimeType": "image/jpeg", "data": base64_image_data}}]}]
        ,"config": {"systemInstruction": SYSTEM_PROMPT,"responseMimeType": "application/json","responseSchema": RESPONSE_SCHEMA}
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(GEMINI_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=30)
            response.raise_for_status() 
            result = response.json()
            json_text = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(json_text)

        except requests.exceptions.RequestException as e:
            if attempt + 1 < max_retries: time.sleep(2 ** attempt); continue
            else: raise Exception(f"Gemini API failed after {max_retries} attempts: {e}")
        except Exception as e:
            raise Exception(f"Failed to parse Gemini response or internal error: {e}")

# --- Integrated Price Prediction Endpoint ---

REQUIRED_FEATURES_INTEGRATED = [
    'image_base64', 'Product_Name', 'Region_Name', 'Harvest_Month', 
    'Market_Demand_Index', 'Storage_Cost_Index', 'Pest_Damage_Ratio', 
    'Current_Temperature_C', 'Last_7_Days_Rainfall_MM'
]

@app.route('/predict', methods=['POST'])
def predict_price():
    """Single endpoint for full prediction: Image input -> Grade Simulation/Real Call -> Price."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body."}), 400

    # 1. Validate Feature Presence
    missing = [f for f in REQUIRED_FEATURES_INTEGRATED if f not in data]
    if missing:
        return jsonify({"error": "Missing required input features.", "missing_features": missing}), 400

    # 2. Get Grade from Image
    base64_input = data['image_base64']
    if isinstance(base64_input, dict): base64_string = list(base64_input.keys())[0] if base64_input else ""
    elif isinstance(base64_input, str): base64_string = base64_input
    else: return jsonify({"error": "Invalid format for image_base64 field."}), 400

    clean_base64 = base64_string.split("base64,")[-1] 
    
    try:
        grade_output = internal_call_gemini_api(clean_base64)
    except Exception as e:
        return jsonify({
            "error": "Image Grading Failed.", "details": str(e),
            "solution": "Check your Gemini API Key and network connectivity. The Base64 string may be malformed."
        }), 503

    quality_grade = grade_output.get('quality_grade')
    grade_info = grade_output.get('justification') or ""
    
    # Check for simulation/fallback grade
    if "SIMULATED" in grade_info or quality_grade is None or quality_grade not in GRADE_TO_SCORE_MAP:
        quality_grade = "B"
        quality_score = GRADE_TO_SCORE_MAP.get("B") 
    else:
        quality_score = GRADE_TO_SCORE_MAP.get(quality_grade) 
    
    # 3. Lookups and Score Mapping
    product_code = PRODUCT_NAME_TO_CODE_MAP.get(data['Product_Name'])
    region_code = REGION_NAME_TO_CODE_MAP.get(data['Region_Name'])
    
    if product_code is None or region_code is None:
        return jsonify({"error": "Lookup failed. Check Product Name or Region Name in the mappings."}), 400

    # 4. Prepare input for the model as a list-of-dicts. Convert to DataFrame only when needed.
    model_input_list = [{
        'Product_Name': product_code, 'Harvest_Month': int(data['Harvest_Month']), 'Region_Code': region_code,
        'Quality_Score': quality_score, 'Pest_Damage_Ratio': float(data['Pest_Damage_Ratio']),
        'Market_Demand_Index': float(data['Market_Demand_Index']), 'Storage_Cost_Index': float(data['Storage_Cost_Index']),
        'Current_Temperature_C': float(data['Current_Temperature_C']), 'Last_7_Days_Rainfall_MM': float(data['Last_7_Days_Rainfall_MM']),
    }]

    # 5. Prediction
    try:
        # If not FAST_MODE, convert to pandas DataFrame just before prediction (lazy import).
        if not FAST_MODE:
            import pandas as pd
            model_input = pd.DataFrame(model_input_list)
        else:
            model_input = model_input_list

        prediction = model_pipeline.predict(model_input)[0]
        
        return jsonify({
            "status": "success", "timestamp": datetime.datetime.now().isoformat(),
            "product_name_input": data['Product_Name'], "region_name_input": data['Region_Name'],
            "quality_grade_used": quality_grade, "grade_justification": grade_info,
            "predicted_base_price_inr_per_kg": round(float(prediction), 2),
            "message": "Full prediction calculated (Image analysis SIMULATED for stability)." if not API_KEY else "Full prediction calculated (Image analysis + Regression)."
        }), 200

    except Exception as e:
        app.logger.error(f"Prediction Error: {e}")
        return jsonify({"error": "Prediction failed due to model error.", "details": str(e)}), 500


if __name__ == '__main__':
    # Load model pipeline upon API startup
    try:
        model_pipeline = load_or_train_model()
    except Exception as e:
        print(f"FATAL ERROR: Could not load or train model. Error: {e}")
        sys.exit(1)
        
    print("--- Starting Single Integrated System (Image Analysis SIMULATION) ---")
    if not API_KEY:
        print("WARNING: API Key is missing. Image grade will be SIMULATED.")
        
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)