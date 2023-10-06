# Import various libraries
from datetime import date
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go

# App Design
st.title("Commodity Price Prediction Web App")# Set Commoditys under consideration
st.header("Available Minerals")
st.write("1. Crude Oil - CL=F")
st.write("2. Gold - GC=F")
st.write("3. Silver - SI=F")

# Getting user inputs
symbol = ("CL=F","GC=F", "SI=F") 
selected_commodity = st.selectbox("Select commodity for prediction", symbol) 
min_date = date(2000, 1, 1) 
max_date = date.today() 
start_date = date(2015, 1, 1) 
start = st.date_input("Pick a Start Date", value=start_date, min_value=min_date, max_value=max_date) 
start = start.strftime("%Y-%m-%d") 

end = date.today().strftime("%Y-%m-%d") 

# Set years for prediction input
n_years = st.slider("Pick number of year(s) for prediction:", 1, 5)
period = n_years * 365

# Data retrieval
@st.cache_data
def load_data(symbol):
    data = yf.download(symbol, start, end)
    data.reset_index(inplace = True)
    return data

data = load_data(selected_commodity)

# Show data
st.subheader("Raw Commodity Historical Data")
st.write(data.head())
st.write(data.tail())

# Visualization
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = "commodity_open", line=dict(color='green')))
    fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = "commodity_close", line=dict(color='blue')))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)


st.subheader('Time Series Analysis of Commodity Price')
plot_raw_data()

# Commodity Forecasting
df_train = data[['Date', 'Close']] 
df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"}) 
 
# Make prediction
model = Prophet() 
model.fit(df_train) 
future = model.make_future_dataframe(periods = period) 
forecast = model.predict(future)

# Show forecast data
st.subheader("Forecast data")
st.write(forecast.tail())

#  visualize Forecast Data
st.subheader("Forecast Analysis")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

# Plot Forecast Trends
st.subheader("Forecast Trend Analysis")
fig2 = model.plot_components(forecast)
st.write(fig2)



