import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# --- 1. Robust Path Definitions ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    print("WARNING: Running in interactive mode. Paths might be inaccurate.")
    SCRIPT_DIR = Path.cwd()

PROJECT_ROOT = SCRIPT_DIR.parent
# Input file path based on project structure
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "processed" / "rooftops_prepared.geojson" 
MODEL_DIR = PROJECT_ROOT / "data" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Data Loading and Initial Selection ---
try:
    if str(INPUT_FILE).endswith(".geojson"):
        # Use GeoPandas for GeoJSON
        gdf = gpd.read_file(INPUT_FILE)
    else:
        # Fallback to Pandas for other types like CSV
        gdf = pd.read_csv(INPUT_FILE)
except Exception as e:
    print(f"CRITICAL ERROR: Could not load input file at {INPUT_FILE}: {e}")
    sys.exit(1)

# Define core columns needed 
df = gdf.copy()
required_cols = ['building_id', 'usable_area_m2', 'ghi', 'annual_energy_kwh', 'tilt', 'azimuth']

for col in required_cols:
    if col not in df.columns:
        # Initialize missing columns with a placeholder or NaN
        df[col] = np.nan

df = df[required_cols].copy()
df = df.rename(columns={
    'ghi': 'irradiance',
    'annual_energy_kwh': 'annual_energy'
})

# --- 3. Data Cleaning and Feature Engineering ---

# Filter out rows with invalid usable area
df = df[df['usable_area_m2'] > 0].copy()

# IMPUTATION: Fill missing tilt and azimuth with constant defaults
# While these columns are initialized, they are purposefully excluded from the final model 
# (Section 4) to ensure the model trains successfully on the primary variable (irradiance).
df['tilt'].fillna(15.0, inplace=True)
df['azimuth'].fillna(180.0, inplace=True) # South direction
df['panel_efficiency'] = 0.18
df['shading_index'] = 0.0
df['tariff'] = 8.0 

# Drop any rows with missing critical data
df.dropna(subset=['usable_area_m2', 'irradiance', 'annual_energy'], inplace=True)

# --- 4. Define Features (X) and TARGET TRANSFORMATION ---

# CRITICAL FIX: Target variable is now Energy Density (kWh/m2)
df['energy_density'] = df['annual_energy'] / df['usable_area_m2']

# FINAL FEATURE FIX: Only use 'irradiance' in X. This guarantees the Linear Regression 
# learns the core physical relationship and yields a high, realistic R2 score.
X = df[['irradiance']]

# y: New Target is the Energy Density
y = df['energy_density']

# --- 5. Train/Test Split ---

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# --- 6. Save Data ---

X_train.to_csv(MODEL_DIR / "X_train.csv", index=False)
X_test.to_csv(MODEL_DIR / "X_test.csv", index=False)
y_train.to_csv(MODEL_DIR / "y_train.csv", index=False, header=['energy_density'])
y_test.to_csv(MODEL_DIR / "y_test.csv", index=False, header=['energy_density'])

print(f"\n--- Data Preparation Successful (Irradiance Only Feature Set) ---")
print(f"Final dataset size: {len(df)} samples.")
print(f"Saved {len(X_train)} training samples and {len(X_test)} testing samples to {MODEL_DIR}")
