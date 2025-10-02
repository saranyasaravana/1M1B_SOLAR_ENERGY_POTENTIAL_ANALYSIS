




import pandas as pd
import joblib 
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# --- PATH SETUP (Guaranteed Reliability) ---

# Get the directory where the current script resides (scripts folder)
SCRIPT_DIR = Path(__file__).resolve().parent

# Define the project root directory (one level above 'scripts')
PROJECT_ROOT = SCRIPT_DIR.parent 

# Define input and output paths
X_TRAIN_PATH = PROJECT_ROOT / "data" / "model" / "X_train.csv"
Y_TRAIN_PATH = PROJECT_ROOT / "data" / "model" / "y_train.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "solar_model_gbr.pkl"

# Ensure the models directory exists before saving
MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def train_and_save_model():
    """Trains a Gradient Boosting Regressor model and saves it to a .pkl file."""
    try:
        # 1. Load Training Data
        print("1. Loading training data...")
        X_train = pd.read_csv(X_TRAIN_PATH)
        y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0] # Read target variable (first column)
        
        print(f"Data loaded successfully. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: Training data not found. Please ensure 'X_train.csv' and 'y_train.csv' are in the 'data/model' folder.")
        print(f"Missing file: {e.filename}")
        return
    except Exception as e:
        print(f"\nFATAL ERROR during data loading: {e}")
        return

    # 2. Define and Tune Model (Gradient Boosting Regressor)
    print("\n2. Starting model training (Gradient Boosting Regressor)...")
    
    # Simple tuning grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    
    gbr = GradientBoostingRegressor(random_state=42)
    
    # Use Grid Search for basic hyperparameter tuning
    grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    final_model = grid_search.best_estimator_
    print(f"Training complete. Best parameters found: {grid_search.best_params_}")

    # 3. Save the Trained Model
    print(f"\n3. Saving the final trained model to: {MODEL_OUTPUT_PATH}")
    joblib.dump(final_model, MODEL_OUTPUT_PATH)
    print("Model saved successfully! You can now proceed to prediction (Step 5).")

if __name__ == "__main__":
    train_and_save_model()
