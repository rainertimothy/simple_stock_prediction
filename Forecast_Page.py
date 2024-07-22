import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yfinance as yf
import streamlit as st
from datetime import date

from sklearn.model_selection import train_test_split

import streamlit as st
import yfinance as yf
from datetime import date

st.set_page_config(
    page_title="Stock Forecast",
    page_icon="📈",
)

st.title('Stock Forecast App with Linear Regression')

st.sidebar.header('Stock Forecast')

START = st.date_input("Enter the starting date:")

TODAY = date.today().strftime("%Y-%m-%d")

st.text("You can check for list in Stock List page")

user_input = st.text_input("Enter a stock ticker (add .JK into Indonesia stock):", 'AAPL') 

if st.button('Predict'):
    @st.cache_data()
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Displaying data...')
    data = load_data(user_input)
    data_load_state.text('Done!')
    data.set_index(data.columns[0], inplace=True)

    st.subheader('Raw data')
    st.write(data)
else:
    st.write("Enter the details and click 'Predict'.")

def plot_raw_data():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Stock Price')
    st.pyplot(fig)

st.subheader(f'{user_input} Price')
plot_raw_data()

st.header('Forecast')

def create_train_test_set(data):
    
    features = data.drop(columns=['Close'], axis=1)
    target = data['Close']
    
    data_len = data.shape[0]
    print("Stock Data length: ",str(data_len))

    train_size = int(data_len * 0.7)
    print("Training Set length: ",str(train_size))

    test_size = int(data_len * 0.3)
    test_index = train_size + test_size
    print("Test Set length: ",str(test_size))

    X_train, X_test = features[:train_size], features[train_size:test_index]
    y_train, y_test = target[:train_size], target[train_size:test_index]

    print(X_train.shape, X_test.shape)
    print(y_train.shape, y_test.shape)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = create_train_test_set(data)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

df_pred = pd.DataFrame(y_test.values, columns=['Actual'], index=y_test.index)
df_pred['Predicted'] = y_pred
df_pred = df_pred.reset_index()
df_pred.loc[:, 'Date'] = pd.to_datetime(df_pred['Date'],format='%Y-%m-%d')

st.subheader('Dataset Actual vs Predicted')
st.write(df_pred)

st.subheader('Actual vs Predicted Stock Price')

fig, ax = plt.subplots(figsize=(12, 6))
sns.set_style("whitegrid")
ax.plot(df_pred['Date'], df_pred['Actual'], label='Actual')
ax.plot(df_pred['Date'], df_pred['Predicted'], label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.legend(loc='best')
ax.grid(True)
st.pyplot(fig)