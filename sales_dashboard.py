import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv("sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Sidebar - Filter
st.sidebar.header("Filter Data")

# Region filter
region = st.sidebar.multiselect(
    "Select Region:",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

# Date filter
min_date = df["Date"].min()
max_date = df["Date"].max()

start_date, end_date = st.sidebar.date_input(
    "Select Date Range:",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Apply filters
filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
]

# Dashboard title
st.title("📊 Sales Dashboard")

# KPIs


# KPI calculations (place this after filtering the DataFrame)
total_revenue = filtered_df["Revenue"].sum()
total_units = filtered_df["Units Sold"].sum()

# Layout: KPIs in columns
st.markdown("### Key Metrics")
kpi1, kpi2 = st.columns(2)

with kpi1:
    st.metric("💰 Total Revenue", f"${total_revenue:,}")
    st.markdown(
        f"""
        <div style="background-color:#e6f4ea; padding:10px; border-radius:10px">
            <strong style="color:#228B22;">+ Revenue is growing!</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi2:
    st.metric("📦 Units Sold", f"{total_units:,}")
    st.markdown(
        f"""
        <div style="background-color:#f0f4ff; padding:10px; border-radius:10px">
            <strong style="color:#1E90FF;">Solid unit sales!</strong>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("### 📊 Sales Distribution by Region")

# Pie Chart: Revenue share by Region
fig_pie = px.pie(
    filtered_df,
    names='Region',
    values='Revenue',
    title='Revenue Share by Region',
    color_discrete_sequence=px.colors.qualitative.Pastel
)
st.plotly_chart(fig_pie, use_container_width=True)


# Revenue over time with prediction
st.subheader("Revenue Over Time (with Forecast)")

from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare actual data
revenue_trend = filtered_df.groupby("Date")["Revenue"].sum().reset_index()
revenue_trend = revenue_trend.sort_values("Date")

# Encode dates as integers for regression
revenue_trend["Date_Ordinal"] = revenue_trend["Date"].map(pd.Timestamp.toordinal)

# Train model
X = revenue_trend["Date_Ordinal"].values.reshape(-1, 1)
y = revenue_trend["Revenue"].values
model = LinearRegression()
model.fit(X, y)

# Predict next 30 days
future_dates = pd.date_range(revenue_trend["Date"].max(), periods=30, freq="D")
future_ordinals = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
future_predictions = model.predict(future_ordinals)

# Build forecast dataframe
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Revenue": future_predictions
})

# Combine actual + forecast
combined_df = pd.concat([revenue_trend[["Date", "Revenue"]], forecast_df])

# Plot
fig_forecast = px.line(combined_df, x="Date", y="Revenue", title="Revenue Forecast")
fig_forecast.add_scatter(x=revenue_trend["Date"], y=revenue_trend["Revenue"], mode="lines+markers", name="Actual")
fig_forecast.add_scatter(x=forecast_df["Date"], y=forecast_df["Revenue"], mode="lines", name="Forecast", line=dict(dash="dash"))

st.plotly_chart(fig_forecast)


# Revenue by product
st.subheader("Revenue by Product")
revenue_product = filtered_df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).reset_index()
fig2 = px.bar(revenue_product, x="Product", y="Revenue", color="Product")
st.plotly_chart(fig2)

# Raw data
with st.expander("Show Raw Data"):
    st.dataframe(filtered_df)
# 📤 Export to CSV
st.markdown("### 📤 Export Filtered Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='filtered_sales_data.csv',
    mime='text/csv',
    help="Click to download the filtered table as a .csv file"
)
