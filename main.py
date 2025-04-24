# main.py
# Aggregate to yearly averages using pandas if wanted
#yearly_df = (
#    monthly_df.groupby(["state_name", pd.Grouper(key="period", freq="Y")])
#    ["value"].mean()
#    .reset_index()
#    .rename(columns={"period": "year"})
#)
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# API URL and Key
API_URL = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
API_KEY = "RQ2OgB4gB1KqEsHstWeCpRasybpciFSW8NHMZXWn"

# Function to fetch data from API
def fetch_data(api_url, api_key):
    params = {
        "api_key": api_key,
        "frequency": "monthly",
        "data[0]": "value",
        "start": "2010-01",
        "end": "2023-01",
        "sort[0][column]": "period",
        "sort[0][direction]": "asc"
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data['response']['data'])
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

# Function to preprocess data
def preprocess_data(df):
    # Convert period to datetime
    df['period'] = pd.to_datetime(df['period'])
    
    # Extract year and month for feature engineering
    df['year'] = df['period'].dt.year
    df['month'] = df['period'].dt.month
    
    # Drop unnecessary columns
    df = df.drop(columns=['duoarea', 'area-name', 'product', 'process', 'series', 'series-description'])
    
    # Handle missing values
    df['value'] = df['value'].replace('', np.nan)
    df['value'] = df['value'].astype(float)
    df['value'].fillna(df['value'].median(), inplace=True)
    
    # Sort by period
    df = df.sort_values(by='period')
    
    return df

# Function to train and evaluate the model
def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title("Actual vs Predicted Petroleum Prices")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    return model

# Main function
def main():
    # Fetch data from API
    print("Fetching data from API...")
    df = fetch_data(API_URL, API_KEY)
    
    # Preprocess data
    print("Preprocessing data...")
    df = preprocess_data(df)
    print(df.head())

    # Feature engineering
    X = df[['year', 'month']]
    y = df['value']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate the model
    print("Training model...")
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Save the model for future use
    joblib.dump(model, 'petroleum_price_model.pkl')
    print("Model saved to disk.")

if __name__ == "__main__":
    main()