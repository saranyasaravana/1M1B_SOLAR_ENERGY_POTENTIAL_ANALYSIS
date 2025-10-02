import pandas as pd
import geopandas as gpd
import joblib 
import numpy as np
from pathlib import Path

# --- CONFIGURATION & ASSUMPTIONS ---
# These must match the columns used in scripts/train_model.py
FEATURE_COLUMNS = ['irradiance'] # Only irradiance is used after model correction

# Financial/Physical Parameters
ANNUAL_YIELD_FACTOR_KWH_PER_KW = 1200  # kWh produced per 1 kWp installed capacity
ELECTRICITY_COST_PER_KWH = 8.0         # Cost per kWh for savings calculation (e.g., ₹8.00 / kWh)
AVG_IRRADIANCE_FACTOR = 5200.0         # Base irradiance used for training simulation

# --- FILE PATH SETUP (EASILY CONFIGURABLE) ---

# Get the directory where the current script resides (scripts folder)
SCRIPT_DIR = Path(__file__).resolve().parent
# Define the project root directory (one level above 'scripts')
PROJECT_ROOT = SCRIPT_DIR.parent 

# --- INPUT CONFIGURATION (MATCHING THE 'prepare_dataset.py' STRUCTURE) ---
# Setting the path to data/raw/processed/rooftops_prepared.geojson
INPUT_BASE_FOLDER = "raw" 
INPUT_SUB_FOLDER = "processed" 
INPUT_FILENAME = "rooftops_prepared.geojson" 

# Dynamic Input Path Construction
INPUT_GEOJSON_PATH = PROJECT_ROOT / "data" / INPUT_BASE_FOLDER / INPUT_SUB_FOLDER / INPUT_FILENAME
# Clean up path in case SUB_FOLDER is empty
INPUT_GEOJSON_PATH = Path(str(INPUT_GEOJSON_PATH).replace("//", "/"))

# Model Input Path 
MODEL_PATH = PROJECT_ROOT / "models" / "solar_model_gbr.pkl"

# OUTPUTS 
OUTPUT_GEOJSON_PATH = PROJECT_ROOT / "data" / "outputs" / "rooftops_with_predictions.geojson"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "outputs" / "rooftops_for_tableau.csv"


def predict_and_export_results():
    """Loads the model and data, predicts energy, calculates metrics, and exports files."""
    try:
        # 1. Load Data and Model
        print("1. Loading data and trained model...")
        
        # Check if the model exists first 
        if not MODEL_PATH.exists():
             raise FileNotFoundError(f"Model file not found. Expected: {MODEL_PATH}")
            
        gdf = gpd.read_file(INPUT_GEOJSON_PATH) 
        gbr = joblib.load(MODEL_PATH)
        
        # Rename column to match expected feature name
        gdf = gdf.rename(columns={'ghi': 'irradiance'}, errors='ignore')
        
        # Ensure 'usable_area_m2' exists and is not zero
        gdf = gdf[gdf['usable_area_m2'] > 0].copy()
        
        print(f"Data and model loaded successfully. Predicting for {len(gdf)} rooftops.")
        
    except FileNotFoundError as e:
        # Displays a clear message to the user about which file is missing
        print(f"\nFATAL ERROR: A required file was not found.")
        print(f"Attempted to access: {e.filename}")
        print("\n--- DEBUG CHECKLIST ---")
        print(f"1. Is the model 'solar_model_gbr.pkl' inside the 'models' folder?")
        print(f"2. Is the GeoJSON file named '{INPUT_FILENAME}' in the directory: 'data/{INPUT_BASE_FOLDER}/{INPUT_SUB_FOLDER}'?")
        return
    except Exception as e:
        print(f"\nFATAL ERROR: An error occurred during data loading or setup: {e}")
        return

    # --- 2. Prepare Feature Data for Prediction (Replicating Training Simulation) ---
    print("\n2. Preparing feature data (Simulating Irradiance Variance)...")
    
    # Replicate the simulated training data: a column of irradiance with variance, 
    # to match the model's training method.
    X_all_data = np.random.normal(loc=AVG_IRRADIANCE_FACTOR, scale=AVG_IRRADIANCE_FACTOR * 0.0001, size=len(gdf))
    X_all = pd.DataFrame(X_all_data, columns=['irradiance'], index=gdf.index)

    # --- 3. Prediction and Calculation ---
    print("\n3. Predicting and calculating metrics...")
    
    # Predict energy density (kWh/m2)
    gdf['pred_energy_density_kwh_m2'] = gbr.predict(X_all)
    gdf['pred_energy_density_kwh_m2'] = np.maximum(0, gdf['pred_energy_density_kwh_m2']) 
    
    # Final Calculations
    gdf['pred_annual_kwh'] = gdf['pred_energy_density_kwh_m2'] * gdf['usable_area_m2']
    gdf['pred_capacity_kW'] = gdf['pred_annual_kwh'] / ANNUAL_YIELD_FACTOR_KWH_PER_KW
    gdf['annual_savings'] = gdf['pred_annual_kwh'] * ELECTRICITY_COST_PER_KWH

    # --- 4. Prepare for Export ---
    
    # Calculate Centroids for lat/lon 
    if 'lat' not in gdf.columns or 'lon' not in gdf.columns:
        print("4. Calculating building centroids for lat/lon...")
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        centroids = gdf_wgs84.geometry.centroid
        gdf['lon'] = centroids.x
        gdf['lat'] = centroids.y
        
    # Create a stable ID for the CSV export
    if 'building_id' not in gdf.columns:
        gdf['building_id'] = gdf.index.astype(str)

    # --- 5. Export Deliverables ---

    # Export GeoJSON Deliverable
    print(f"\n5. Saving GeoJSON with predictions to: {OUTPUT_GEOJSON_PATH.name}")
    gdf.to_file(OUTPUT_GEOJSON_PATH, driver="GeoJSON")
    print("GeoJSON saved successfully.")


    # Export CSV for Tableau
    print("6. Preparing and saving CSV export for Tableau...")

    csv_data = gdf[['building_id', 'lat', 'lon', 'usable_area_m2', 'irradiance', 
                    'pred_annual_kwh', 'pred_capacity_kW', 'annual_savings']].copy()

    # Rename columns for clarity in Tableau
    csv_data = csv_data.rename(columns={
        'building_id': 'Building_ID', 
        'usable_area_m2': 'Usable_Area_m2', 
        'irradiance': 'Irradiance_W_m2',
        'pred_annual_kwh': 'Predicted_Annual_kWh',
        'pred_capacity_kW': 'Predicted_Capacity_kW',
        'annual_savings': 'Estimated_Annual_Savings_INR' 
    })

    csv_data.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"CSV saved successfully to: {OUTPUT_CSV_PATH.name}")

    print("\n--- Step 5 (Prediction & Export) complete! ---")

if __name__ == "__main__":
    predict_and_export_results()
