# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------
# Helper functions
# ---------------------------
def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred)/(y_true + 1e-9)))*100
def risk_from_fri(fri):
    if fri >= 0.9: return "Low"
    elif fri >= 0.7: return "Medium"
    else: return "High"

# ---------------------------
# Page config + CSS
# ---------------------------
st.set_page_config(page_title="Flood Resilience Dashboard", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background: linear-gradient(180deg, #f8fafc, #000000);}
h1 { font-weight: 700; color: white; }
.risk-low { color: #065f46; font-weight:700; }
.risk-med { color: #b45309; font-weight:700; }
.risk-high { color: #7f1d1d; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Flood Resilience Dashboard — Division-level Crop Yield")
st.write("Compute FSIs, FRI, predict yields (Ridge / RF / CatBoost), and explore SHAP explanations.")

# ---------------------------
# Load dataset via file uploader
# ---------------------------
st.sidebar.markdown("## Upload dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded_file is None:
    st.warning("Please upload your CSV to continue.")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

df = load_data(uploaded_file)

st.sidebar.markdown("## Dataset overview")
st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
if st.sidebar.checkbox("Show raw data (first 10 rows)"):
    st.dataframe(df.head(10))

# ---------------------------
# Ensure required columns
# ---------------------------
TARGET = "Crop yield"
REQUIRED_COLS = [
    "Division","year","hectares", "Precipitation Corrected Sum",
    "Root Zone Soil Wetness", "Surface Soil Wetness", "Max temp Avg",
    "Max Wind Speed", "Humidity", "Earth Skin Temp"
]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Fill missing numeric columns
df = df.copy()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ---------------------------
# Feature engineering: FSIs
# ---------------------------
st.markdown("### 🔧 Feature engineering (FSIs)")
df["precip_mean_div"] = df.groupby("Division")["Precipitation Corrected Sum"].transform("mean")
df["precip_std_div"] = df.groupby("Division")["Precipitation Corrected Sum"].transform("std").replace(0,1e-6)
df["FSI1_precip_anom"] = (df["Precipitation Corrected Sum"] - df["precip_mean_div"]) / df["precip_std_div"]

thresh_root = df["Root Zone Soil Wetness"].quantile(0.90)
mean_max_wind = df["Max Wind Speed"].mean() + 1e-6

df["FSI2_soil_excess"] = np.maximum(0, df["Root Zone Soil Wetness"] - thresh_root)
df["FSI3_precip_per_ha"] = df["Precipitation Corrected Sum"] / (df["hectares"] + 1e-6)
df["FSI4_wind_norm"] = df["Max Wind Speed"] / mean_max_wind
df["FSI5_temp_precip"] = df["Max temp Avg"] * df["Precipitation Corrected Sum"]
df["FSI6_hum_surface"] = df["Humidity"] * df["Surface Soil Wetness"]

df = df.sort_values(["Division","year"])
df["Yield_lag1"] = df.groupby("Division")[TARGET].shift(1)
df["FSI1_lag1"] = df.groupby("Division")["FSI1_precip_anom"].shift(1)
df = df.fillna(method="bfill").fillna(0)
st.success("FSIs computed.")

# ---------------------------
# Expected Yield & FRI
# ---------------------------
st.markdown("### 🔎 Compute Expected Yield & FRI")
features_for_expected = ["FSI1_precip_anom","FSI2_soil_excess","FSI3_precip_per_ha","Yield_lag1"]
mask_non_extreme = df["FSI1_precip_anom"].abs() <= 2.0
X_exp = df.loc[mask_non_extreme, features_for_expected]
y_exp = df.loc[mask_non_extreme, TARGET]

if len(X_exp) >= 10:
    ridge_exp = Ridge(alpha=1.0)
    ridge_exp.fit(X_exp, y_exp)
    df["Expected_Yield"] = ridge_exp.predict(df[features_for_expected])
else:
    df["Expected_Yield"] = df[TARGET].rolling(3,min_periods=1).mean()

df["FRI"] = df[TARGET] / (df["Expected_Yield"] + 1e-6)
df["Risk_Category"] = df["FRI"].apply(risk_from_fri)
st.success("Expected Yield & FRI computed.")

# ---------------------------
# Modeling
# ---------------------------
st.markdown("### 🧠 Train models (Ridge / RandomForest / CatBoost)")
feature_cols = [
    "FSI1_precip_anom","FSI2_soil_excess","FSI3_precip_per_ha",
    "FSI4_wind_norm","FSI5_temp_precip","FSI6_hum_surface",
    "Yield_lag1","FSI1_lag1"
]

X = df[feature_cols]
y = df[TARGET]
df = df.sort_values(["Division","year"])

test_years_per_div = df.groupby("Division")["year"].apply(lambda x: x.nlargest(max(1,int(0.2*len(x))))).explode()
test_mask = df.index.isin(test_years_per_div.index)
train_mask = ~test_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

if len(y_test) < 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

cat = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, verbose=False, random_seed=42)
cat.fit(X_train, y_train)
cat_pred = cat.predict(X_test)

# ---------------------------
# Evaluation
# ---------------------------
def eval_table(y_true, preds, model_name):
    return {
        "model": model_name,
        "RMSE": rmse(y_true, preds),
        "MAE": mean_absolute_error(y_true, preds),
        "R2": r2_score(y_true, preds),
        "MAPE(%)": mape(y_true, preds)
    }

evals = [
    eval_table(y_test, ridge_pred, "Ridge"),
    eval_table(y_test, rf_pred, "RandomForest"),
    eval_table(y_test, cat_pred, "CatBoost"),
]

eval_df = pd.DataFrame(evals).set_index("model")
st.dataframe(eval_df.style.format("{:.3f}"))

# ---------------------------
# SHAP explainability
# ---------------------------
st.markdown("### 🔍 SHAP explainability (CatBoost)")
explainer = shap.TreeExplainer(cat)
shap_vals = explainer.shap_values(X_train)

# ---------------------------
# Sidebar: scenario controls
# ---------------------------
st.sidebar.markdown("## Scenario Controls")
divisions = df["Division"].unique().tolist()
sel_div = st.sidebar.selectbox("Division", divisions, index=0)
years_for_div = sorted(df[df["Division"]==sel_div]["year"].unique())
sel_year = st.sidebar.selectbox("Year", years_for_div, index=len(years_for_div)-1)

scenario_row = df[(df["Division"]==sel_div) & (df["year"]==sel_year)]
if scenario_row.empty:
    st.warning("No data for selected division/year.")
    st.stop()
scenario_row = scenario_row.iloc[0].copy()

# Sliders for scenario variables
scenario_row["Precipitation Corrected Sum"] = st.sidebar.slider(
    "Precipitation Corrected Sum",
    float(df["Precipitation Corrected Sum"].min()),
    float(df["Precipitation Corrected Sum"].max()),
    float(scenario_row["Precipitation Corrected Sum"])
)
scenario_row["Root Zone Soil Wetness"] = st.sidebar.slider(
    "Root Zone Soil Wetness",
    float(df["Root Zone Soil Wetness"].min()),
    float(df["Root Zone Soil Wetness"].max()),
    float(scenario_row["Root Zone Soil Wetness"])
)
scenario_row["Surface Soil Wetness"] = st.sidebar.slider(
    "Surface Soil Wetness",
    float(df["Surface Soil Wetness"].min()),
    float(df["Surface Soil Wetness"].max()),
    float(scenario_row["Surface Soil Wetness"])
)
scenario_row["Max temp Avg"] = st.sidebar.slider(
    "Max Temp Avg",
    float(df["Max temp Avg"].min()),
    float(df["Max temp Avg"].max()),
    float(scenario_row["Max temp Avg"])
)
scenario_row["Max Wind Speed"] = st.sidebar.slider(
    "Max Wind Speed",
    float(df["Max Wind Speed"].min()),
    float(df["Max Wind Speed"].max()),
    float(scenario_row["Max Wind Speed"])
)
scenario_row["Humidity"] = st.sidebar.slider(
    "Humidity",
    float(df["Humidity"].min()),
    float(df["Humidity"].max()),
    float(scenario_row["Humidity"])
)
scenario_row["hectares"] = st.sidebar.slider(
    "Hectares",
    float(df["hectares"].min()),
    float(df["hectares"].max()),
    float(scenario_row["hectares"])
)

# Recompute scenario FSIs
scenario_row["FSI1_precip_anom"] = (scenario_row["Precipitation Corrected Sum"] - df[df["Division"]==sel_div]["Precipitation Corrected Sum"].mean()) / df[df["Division"]==sel_div]["Precipitation Corrected Sum"].std()
scenario_row["FSI2_soil_excess"] = max(0, scenario_row["Root Zone Soil Wetness"] - thresh_root)
scenario_row["FSI3_precip_per_ha"] = scenario_row["Precipitation Corrected Sum"] / (scenario_row["hectares"] + 1e-6)
scenario_row["FSI4_wind_norm"] = scenario_row["Max Wind Speed"] / mean_max_wind
scenario_row["FSI5_temp_precip"] = scenario_row["Max temp Avg"] * scenario_row["Precipitation Corrected Sum"]
scenario_row["FSI6_hum_surface"] = scenario_row["Humidity"] * scenario_row["Surface Soil Wetness"]

scenario_features = feature_cols
scenario_input = pd.DataFrame([scenario_row[scenario_features]])

# Scenario predictions
st.markdown(f"### 📊 Scenario predictions — {sel_div} / {sel_year}")
ridge_scenario = ridge.predict(scenario_input)[0]
rf_scenario = rf.predict(scenario_input)[0]
cat_scenario = cat.predict(scenario_input)[0]

st.write(f"**Ridge predicted yield:** {ridge_scenario:.2f}")
st.write(f"**RandomForest predicted yield:** {rf_scenario:.2f}")
st.write(f"**CatBoost predicted yield:** {cat_scenario:.2f}")

expected_yield_scenario = scenario_row["Expected_Yield"]
fri_scenario = scenario_row[TARGET] / (expected_yield_scenario + 1e-6)
risk_scenario = risk_from_fri(fri_scenario)
st.write(f"**FRI:** {fri_scenario:.2f} → **Risk Category:** {risk_scenario}")

# SHAP plot
st.markdown("### 🌟 SHAP plot (CatBoost)")
fig, ax = plt.subplots(figsize=(8,5))
shap.summary_plot(shap_vals, X_train, plot_type="bar", show=False)
st.pyplot(fig)

# Download predictions
st.markdown("### 💾 Download predictions")
df_preds = df.copy()
df_preds["Ridge_pred"] = ridge.predict(X)
df_preds["RF_pred"] = rf.predict(X)
df_preds["CatBoost_pred"] = cat.predict(X)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(df_preds)
st.download_button(
    label="Download full predictions as CSV",
    data=csv_data,
    file_name=f"flood_yield_preds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

st.success("✅ Streamlit Flood Resilience Dashboard ready!")
