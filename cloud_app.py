import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Cloud AI System", layout="wide")

st.title("☁️ AI Cloud Cost Optimization System")

# -------------------------------
# FILE UPLOAD
# -------------------------------
st.sidebar.header("📂 Upload Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# -------------------------------
# DEFAULT DATA
# -------------------------------
default_data = {
    "Service": ["VM","Storage","Database","VM","Storage","Database"],
    "Usage (%)": [80,20,75,10,15,90],
    "Cost ($)": [5000,3000,4000,1000,1200,6000],
    "Date": [
        "2026-01-01","2026-01-01","2026-01-01",
        "2026-02-01","2026-02-01","2026-02-01"
    ]
}

# -------------------------------
# LOAD DATA
# -------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame(default_data)

df.columns = df.columns.str.strip()

# -------------------------------
# DATE HANDLING
# -------------------------------
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# -------------------------------
# REQUIRED COLUMNS CHECK
# -------------------------------
required_cols = ["Service", "Usage (%)", "Cost ($)"]

if not all(col in df.columns for col in required_cols):
    st.error("CSV must contain: Service, Usage (%), Cost ($)")
    st.stop()

# -------------------------------
# FILTERS
# -------------------------------
st.sidebar.header("🔍 Filters")

service_filter = st.sidebar.multiselect(
    "Service",
    df["Service"].unique(),
    default=df["Service"].unique()
)

df = df[df["Service"].isin(service_filter)]

# -------------------------------
# ANOMALY DETECTION
# -------------------------------
if "Date" in df.columns:
    df = df.sort_values(by=["Service", "Date"])

df["Prev Cost"] = df.groupby("Service")["Cost ($)"].shift(1)

df["Cost Change %"] = (
    (df["Cost ($)"] - df["Prev Cost"]) / df["Prev Cost"]
) * 100

df["Anomaly"] = df["Cost Change %"].apply(
    lambda x: "Yes" if pd.notnull(x) and x > 30 else "No"
)

# -------------------------------
# RISK SCORE
# -------------------------------
df["Risk Score"] = df["Cost ($)"] / (df["Usage (%)"] + 1)

# -------------------------------
# AI RECOMMENDATION
# -------------------------------
def recommend(row):
    if row["Usage (%)"] < 30:
        return "Reduce resource"
    elif row["Cost ($)"] > 6000:
        return "Optimize cost"
    elif row["Anomaly"] == "Yes":
        return "Investigate spike"
    else:
        return "Normal"

df["Recommendation"] = df.apply(recommend, axis=1)

# -------------------------------
# MACHINE LEARNING
# -------------------------------
le = LabelEncoder()
df["Service_Enc"] = le.fit_transform(df["Service"])

X = df[["Service_Enc", "Usage (%)"]]
y = df["Cost ($)"]

model = RandomForestRegressor()
model.fit(X, y)

# -------------------------------
# KPIs
# -------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Cost", f"${df['Cost ($)'].sum():,.0f}")
col2.metric("Avg Usage", f"{df['Usage (%)'].mean():.1f}%")
col3.metric("Max Cost", f"${df['Cost ($)'].max():,.0f}")

# -------------------------------
# TABLE
# -------------------------------
st.subheader("📋 Data")
st.dataframe(df)

# -------------------------------
# CHARTS
# -------------------------------
st.subheader("📈 Trends")

if "Date" in df.columns:
    cost_trend = df.groupby("Date")["Cost ($)"].sum()
    st.line_chart(cost_trend)

cost_service = df.groupby("Service")["Cost ($)"].sum()
st.bar_chart(cost_service)

# -------------------------------
# RISK HEATMAP (SAFE VERSION)
# -------------------------------
st.subheader("🔥 Risk Heatmap")

heatmap = df.pivot_table(
    values="Risk Score",
    index="Service",
    aggfunc="mean"
)

st.dataframe(heatmap)

# -------------------------------
# PREDICTION VS ACTUAL
# -------------------------------
st.subheader("🔮 Prediction vs Actual")

df["Predicted"] = model.predict(X)

st.line_chart(df[["Cost ($)", "Predicted"]])

# -------------------------------
# DRILL DOWN
# -------------------------------
st.subheader("🔍 Drill Down")

service_selected = st.selectbox(
    "Select Service",
    df["Service"].unique()
)

filtered = df[df["Service"] == service_selected]

st.dataframe(filtered)

# -------------------------------
# PREDICTION TOOL
# -------------------------------
st.subheader("🔮 Predict Cost")

svc = st.selectbox("Service Input", df["Service"].unique())
usage = st.slider("Usage (%)", 0, 100, 50)

if st.button("Predict Cost"):
    enc = le.transform([svc])[0]
    pred = model.predict([[enc, usage]])
    st.success(f"Predicted Cost: ${pred[0]:,.0f}")

# -------------------------------
# AI REPORT
# -------------------------------
st.subheader("🧠 AI Report")

if st.button("Generate Report"):
    report = f"""
    Total Resources: {len(df)}
    Anomalies: {len(df[df['Anomaly']=='Yes'])}

    Insights:
    - Underutilized resources detected
    - Cost optimization possible
    - Monitor high-risk services
    """
    st.write(report)