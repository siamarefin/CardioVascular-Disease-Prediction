import pandas as pd
import os 
import matplotlib
matplotlib.use('Agg')

import joblib  # For saving the model



# Get the absolute path to the model file
model_path = os.path.join("XGBoostClassifier_model.pkl")

# Load the model
try:
    best_model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    raise Exception(f"Model file not found at: {model_path}")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

def predict(input_data: dict):
    """
    Predict the outcome based on input data using the pre-trained model.
    :param input_data: Dictionary containing input features.
    :return: Dictionary with prediction and probabilities.
    """
    # Convert JSON input to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure all required columns are present
    required_columns = ["ap_hi", "ap_lo", "cholesterol", "age_years", "bmi"]
    for col in required_columns:
        if col not in input_df.columns:
            return {"error": f"Missing required field: {col}"}

    try:
        # Normalize the input data using the same scaler as in training

        # Make predictions
        prediction = best_model.predict(input_df)

        # Return the result as a dictionary
        return {
            "input": input_data,
            "predicted_cardio": int(prediction[0]),  # 0: No, 1: Yes
        }
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}
