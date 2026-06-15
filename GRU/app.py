import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

st.set_page_config(
    page_title="GRU Forecasting App",
    page_icon="📊",
    layout="wide"
)

st.title("📊 GRU-Based Time Series Forecasting")

st.write("""
Upload a dataset with a single numeric column (Stock/Sales/Price).
This app uses a GRU model to learn patterns and predict future values.
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    data = df[col].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    def create_sequences(data, step=10):
        X, y = [], []
        for i in range(len(data) - step):
            X.append(data[i:i+step])
            y.append(data[i+step])
        return np.array(X), np.array(y)

    step = 10
    X, y = create_sequences(data_scaled, step)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(step, 1)),
        GRU(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    with st.spinner("Training GRU model..."):
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    st.success("Model trained successfully!")

    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds)
    y_test_actual = scaler.inverse_transform(y_test)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=y_test_actual.flatten(),
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        y=preds.flatten(),
        name="Predicted"
    ))

    st.subheader("Prediction vs Actual")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Future Prediction")

    last_seq = data_scaled[-step:].reshape(1, step, 1)

    future = []
    for _ in range(20):
        pred = model.predict(last_seq, verbose=0)
        future.append(pred[0][0])
        last_seq = np.append(last_seq[:, 1:, :], [[pred]], axis=1)

    future = scaler.inverse_transform(np.array(future).reshape(-1, 1))

    st.line_chart(future)