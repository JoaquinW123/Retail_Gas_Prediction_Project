import joblib
import pandas as pd

# Load the saved model
model = joblib.load('petroleum_price_model.pkl')

# Example new data (year and month)
new_data = pd.DataFrame({
    'year': [2024, 2025, 2026],
    'month': [1, 2, 4]  # January and February.  If out of bounds ex. 13 , it will just return the same value as 12
})

# Make predictions
predictions = model.predict(new_data)
print("Predicted Petroleum Prices:", predictions)