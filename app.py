import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime, timedelta
import plotly.graph_objs as go


# Load the saved model
with open('temperature_prediction_model_2.pkl', 'rb') as f:
    model = pickle.load(f)

# Load and preprocess the temperature data
data = pd.read_csv('weather_data.csv')
temperature_data = data[['Temperature']].values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
temperature_data_scaled = scaler.fit_transform(temperature_data)

sequence_length = 30

# Function to forecast temperature using the LSTM model
def forecast_temperature_lstm(model, data, steps=5):
    sequence = data[-sequence_length:]
    predictions = []

    for _ in range(steps):
        sequence_reshaped = np.reshape(sequence, (1, sequence.shape[0], sequence.shape[1]))
        next_temp_scaled = model.predict(sequence_reshaped)
        sequence = np.append(sequence, next_temp_scaled, axis=0)
        sequence = sequence[-sequence_length:]
        next_temp = scaler.inverse_transform(next_temp_scaled)
        predictions.append(next_temp[0][0])

    return predictions

# Function to get current temperature from an API (e.g., OpenWeatherMap)
import requests

def get_current_temperature(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        
        # Check if 'main' and 'temp' keys are in the response data
        if 'main' in data and 'temp' in data['main']:
            current_temp = data['main']['temp']
            return current_temp
        else:
            print("Error: 'temp' data not found in the response.")
            print("Response content:", data)
            return None
    else:
        # Print the status code and error message for debugging
        print(f"Error: Unable to fetch data, status code {response.status_code}")
        print("Response content:", response.text)
        return None


# Streamlit UI
st.title("Temperature Forecast 🌤️")

city_name = st.text_input("Enter your city name:")
api_key = "e268b145029eb95caddf4aa16ebe13e8" 

if st.button("Get Current Temperature"):
    if city_name:
        current_temp = get_current_temperature(api_key, city_name)
        current_temp = get_current_temperature(api_key, city_name)

        if current_temp is not None:
            st.write(f"Current Temperature in {city_name}: {current_temp:.2f} °C")
        else:
            st.write(f"Could not fetch the current temperature for {city_name}. Please check the city name or API key.")


if st.button("Generate Forecast"):
    steps = 5
    forecasted_temps = forecast_temperature_lstm(model, temperature_data_scaled, steps=steps)

    #
    # Create an interactive plot with Plotly
    fig = go.Figure()

    # Add actual temperatures to the figure
    fig.add_trace(go.Scatter(
        x=list(range(len(temperature_data))),
        y=scaler.inverse_transform(temperature_data).flatten(),
        mode='lines+markers',
        name='Actual Temperature',
        line=dict(color='blue')
    ))

    # Add forecasted temperatures to the figure
    fig.add_trace(go.Scatter(
        x=list(range(len(temperature_data), len(temperature_data) + steps)),
        y=forecasted_temps,
        mode='lines+markers',
        name='Forecasted Temperature',
        line=dict(color='red')
    ))

    # Update layout for better visuals
    fig.update_layout(
        title='Actual vs Forecasted Temperatures',
        xaxis_title='Time',
        yaxis_title='Temperature (°C)',
        legend=dict(x=0, y=1),
        template='plotly_white'
    )

    # Display the interactive plot in Streamlit
    st.plotly_chart(fig)

    # Display forecasted temperatures with dates
    st.write("Forecasted Temperatures for the Next 5 Days:")
    
    today = datetime.now()
    
    for i, temp in enumerate(forecasted_temps, 1):
        forecast_date = today + timedelta(days=i)  # Calculate date for each forecast day
        st.write(f"{forecast_date.strftime('%Y-%m-%d')}: {temp:.2f} °C")