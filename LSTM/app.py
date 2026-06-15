import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales Forecast Dashboard")

st.write(
    "Upload historical sales data and forecast future sales trends."
)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if len(df.columns) < 2:
        st.error(
            "Dataset should contain Date and Sales columns."
        )
        st.stop()

    date_col = df.columns[0]
    sales_col = df.columns[1]

    df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(date_col)

    df["Day"] = np.arange(len(df))

    X = df[["Day"]]
    y = df[sales_col]

    model = LinearRegression()
    model.fit(X, y)

    future_days = st.slider(
        "Forecast Days",
        7,
        90,
        30
    )

    future = np.arange(
        len(df),
        len(df) + future_days
    ).reshape(-1, 1)

    forecast = model.predict(future)

    future_dates = pd.date_range(
        start=df[date_col].max() + pd.Timedelta(days=1),
        periods=future_days
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast
    })

    st.subheader("Forecast Results")
    st.dataframe(forecast_df)

    actual_fig = px.line(
        df,
        x=date_col,
        y=sales_col,
        title="Historical Sales"
    )

    st.plotly_chart(
        actual_fig,
        use_container_width=True
    )

    forecast_fig = px.line(
        forecast_df,
        x="Date",
        y="Forecast",
        title="Future Forecast"
    )

    st.plotly_chart(
        forecast_fig,
        use_container_width=True
    )

else:

    st.info(
        "Upload a CSV file with Date and Sales columns."
    )