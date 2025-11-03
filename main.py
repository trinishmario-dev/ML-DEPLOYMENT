import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 🎨 Page config
st.set_page_config(page_title="Beverage Sales Dashboard", layout="wide")

# 💅 Custom CSS for modern dark theme
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #f5f5f5;
        }
        .block-container {
            padding: 2rem 3rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #00e6ac !important;
        }
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }
    </style>
""", unsafe_allow_html=True)

# 🧃 Title
st.title("🥤 Modern Beverage Sales Prediction Dashboard")
st.markdown("### Explore interactive visualizations and predict sales using a machine learning model!")

# 📂 Load data
df = pd.read_csv("data/synthetic_beverage_sales_data.csv", low_memory=False)
df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')

# 🔍 Sidebar filters
st.sidebar.header("Filter Options")

# Convert columns to strings and drop nulls to avoid TypeError
region_list = sorted(df['Region'].dropna().astype(str).unique().tolist())
category_list = sorted(df['Category'].dropna().astype(str).unique().tolist())

region = st.sidebar.selectbox("Select Region", ["All"] + region_list)
category = st.sidebar.selectbox("Select Category", ["All"] + category_list)

filtered_df = df.copy()
if region != "All":
    filtered_df = filtered_df[filtered_df['Region'] == region]
if category != "All":
    filtered_df = filtered_df[filtered_df['Category'] == category]

# 🔢 KPI Cards
total_sales = filtered_df['Total_Price'].sum()
avg_discount = filtered_df['Discount'].mean() * 100
top_product = filtered_df.groupby('Product')['Total_Price'].sum().idxmax()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💰 Total Revenue", f"₹{total_sales:,.0f}")
with col2:
    st.metric("💸 Avg Discount", f"{avg_discount:.2f}%")
with col3:
    st.metric("🏆 Top Product", top_product)

# 📊 Dataset Preview
st.subheader("📋 Filtered Data Preview")
st.dataframe(filtered_df.head())

# ----------------------------
# 📈 Modern Visualizations
# ----------------------------
st.header("📉 Interactive Data Visualizations")

col1, col2 = st.columns(2)

# 1️⃣ Total Sales by Product
with col1:
    fig1 = px.bar(filtered_df.groupby('Product')['Total_Price'].sum().reset_index(),
                  x='Total_Price', y='Product',
                  orientation='h',
                  color='Product',
                  title="Total Sales by Product",
                  color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig1, use_container_width=True)

# 2️⃣ Region-wise Sales Distribution
with col2:
    fig2 = px.pie(df, names='Region', values='Total_Price',
                  hole=0.4, title="Region-wise Sales Distribution",
                  color_discrete_sequence=px.colors.sequential.Teal)
    st.plotly_chart(fig2, use_container_width=True)

# 3️⃣ Monthly Sales Trend
st.subheader("📈 Monthly Sales Trend")
monthly_sales = df.groupby(df['Order_Date'].dt.to_period('M'))['Total_Price'].sum().reset_index()
monthly_sales['Order_Date'] = monthly_sales['Order_Date'].astype(str)

fig3 = px.line(monthly_sales, x='Order_Date', y='Total_Price',
               markers=True, title="Monthly Sales Trend",
               color_discrete_sequence=["#00E6AC"])
fig3.update_traces(line=dict(width=4))
st.plotly_chart(fig3, use_container_width=True)

# 4️⃣ Animated Sales Over Time
st.subheader("🎞️ Sales Growth Over Time (Animated)")
fig4 = px.bar(df,
              x='Product', y='Total_Price',
              color='Region',
              animation_frame=df['Order_Date'].dt.strftime("%Y-%m"),
              title="Sales Growth Over Time",
              color_discrete_sequence=px.colors.qualitative.Safe)
st.plotly_chart(fig4, use_container_width=True)

# ----------------------------
# 🤖 Machine Learning Model
# ----------------------------
st.header("🤖 Predict Total Price using Random Forest")

# 🧹 Clean data (remove NaN values in important columns)
df = df.dropna(subset=['Unit_Price', 'Quantity', 'Discount', 'Total_Price']).reset_index(drop=True)

X = df[['Unit_Price', 'Quantity', 'Discount']]
y = df['Total_Price']

# Train/Test Split and Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

st.success(f"✅ Model trained successfully | Mean Absolute Error: {mae:.2f}")

# 🔮 Prediction Section
st.subheader("💡 Predict New Sale")
col1, col2, col3 = st.columns(3)
unit_price = col1.number_input("Unit Price (₹)", min_value=10.0, max_value=200.0, value=50.0)
quantity = col2.number_input("Quantity", min_value=1, max_value=100, value=5)
discount = col3.number_input("Discount (0-0.5)", min_value=0.0, max_value=0.5, value=0.1, step=0.01)

if st.button("🔮 Predict Total Price"):
    pred = model.predict([[unit_price, quantity, discount]])[0]
    st.success(f"Predicted Total Price: ₹{pred:.2f}")

# ----------------------------
# 🧾 Conclusion
# ----------------------------
st.markdown("""
---
### 📘 Summary
- Interactive analytics with Plotly-powered modern visuals.
- Real-time filters for **Region** and **Category**.
- Predictive insights using **Random Forest Regression**.
- Fully deployable on **Streamlit Cloud** for real-world use.
""")
