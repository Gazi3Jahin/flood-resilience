# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import base64
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
st.set_page_config(page_title="Flood Resilience Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background: linear-gradient(180deg, #f8fafc, #000000);}
.card{background: black; padding: 1rem; border-radius: 12px; box-shadow: 0 6px 20px 4px rgba(0,0,0,0.08); margin-bottom: 1rem; color: black;}
h1 { font-weight: 700; }
.small-muted { color: #6b7280; font-size: 0.9rem; }
.risk-low { color: #065f46; font-weight:700; }
.risk-med { color: #b45309; font-weight:700; }
.risk-high { color: #7f1d1d; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.title("🌾 Flood Resilience Dashboard — Division-level Crop Yield")
st.write("Compute FSIs, FRI, predict yields (Ridge / RF / CatBoost), and explore SHAP explanations.")

# ---------------------------
# Load dataset
# ---------------------------
DATA_PATH = r"C:\Users\gazi3\OneDrive\Documents\Flood resilence\All devision Crop Data (2000-2024)-Merged+Normalized - Sheet1 (2).csv"
if not os.path.exists(DATA_PATH):
    st.error(f"CSV not found at {DATA_PATH}. Upload or place your CSV at this path.")
    st.stop()

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

df = load_data(DATA_PATH)
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
with st.spinner("Computing FSIs..."):
    df["precip_mean_div"] = df.groupby("Division")["Precipitation Corrected Sum"].transform("mean")
    df["precip_std_div"] = df.groupby("Division")["Precipitation Corrected Sum"].transform("std").replace(0,1e-6)
    df["FSI1_precip_anom"] = (df["Precipitation Corrected Sum"] - df["precip_mean_div"]) / df["precip_std_div"]

    thresh_root = df["Root Zone Soil Wetness"].quantile(0.90)
    df["FSI2_soil_excess"] = np.maximum(0, df["Root Zone Soil Wetness"] - thresh_root)

    df["FSI3_precip_per_ha"] = df["Precipitation Corrected Sum"] / (df["hectares"] + 1e-6)

    mean_max_wind = df["Max Wind Speed"].mean() + 1e-6
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
# Modeling: Ridge / RF / CatBoost
# ---------------------------
# ---------------------------
# Modeling: prepare X, y
# ---------------------------
st.markdown("### 🧠 Train models (Ridge / RandomForest / CatBoost)")

feature_cols = [
    "FSI1_precip_anom","FSI2_soil_excess","FSI3_precip_per_ha",
    "FSI4_wind_norm","FSI5_temp_precip","FSI6_hum_surface",
    "Yield_lag1","FSI1_lag1"
]

X = df[feature_cols]
y = df[TARGET]
groups = df["Division"]

# ---------------------------
# Division-aware train-test split
# ---------------------------

# Temporal split: last 20% years per division for test
df = df.sort_values(["Division","year"])
test_years_per_div = df.groupby("Division")["year"].apply(lambda x: x.nlargest(max(1,int(0.2*len(x))))).explode()
test_mask = df.index.isin(test_years_per_div.index)
train_mask = ~test_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# If test set too small, fallback to simple random split
if len(y_test) < 10:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Train models
# ---------------------------

# 1) Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

# 2) Random Forest
rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 3) CatBoost
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
# Optional: baseline comparison
# ---------------------------



# ---------------------------
# SHAP explainability
# ---------------------------
st.markdown("### 🔍 SHAP explainability (CatBoost)")
with st.spinner("Computing SHAP values..."):
    explainer = shap.TreeExplainer(cat)
    shap_vals = explainer.shap_values(X_train)

# ---------------------------
# Sidebar: scenario controls
# ---------------------------
st.sidebar.markdown("## Controls")
divisions = df["Division"].unique().tolist()
sel_div = st.sidebar.selectbox("Division", divisions, index=0)
years_for_div = sorted(df[df["Division"]==sel_div]["year"].unique())
sel_year = st.sidebar.selectbox("Year", years_for_div, index=len(years_for_div)-1)

st.sidebar.markdown("### Scenario sliders")
precip_mult = st.sidebar.slider("Precipitation multiplier (%)",50,200,100,5)
soil_add = st.sidebar.slider("Add Root Zone Soil Wetness",0.0,2.0,0.0,0.1)
temp_add = st.sidebar.slider("Add Max Temp Avg (°C)",-3.0,5.0,0.0,0.5)
wind_mult = st.sidebar.slider("Wind multiplier (%)",50,200,100,5)

row = df[(df["Division"]==sel_div)&(df["year"]==sel_year)]
if row.empty: st.error("No data for selected Division & Year"); st.stop()
row = row.iloc[0].copy()
row_s = row.copy()
row_s["Precipitation Corrected Sum"] *= (precip_mult/100.0)
row_s["Root Zone Soil Wetness"] += soil_add
row_s["Max temp Avg"] += temp_add
row_s["Max Wind Speed"] *= (wind_mult/100.0)

# recompute FSIs
row_s["FSI1_precip_anom"] = (row_s["Precipitation Corrected Sum"] - row["precip_mean_div"]) / (row["precip_std_div"]+1e-6)
row_s["FSI2_soil_excess"] = max(0,row_s["Root Zone Soil Wetness"] - thresh_root)
row_s["FSI3_precip_per_ha"] = row_s["Precipitation Corrected Sum"]/(row_s["hectares"]+1e-6)
row_s["FSI4_wind_norm"] = row_s["Max Wind Speed"]/mean_max_wind
row_s["FSI5_temp_precip"] = row_s["Max temp Avg"]*row_s["Precipitation Corrected Sum"]
row_s["FSI6_hum_surface"] = row_s["Humidity"]*row_s["Surface Soil Wetness"]
row_s["Yield_lag1"] = row["Yield_lag1"]
row_s["FSI1_lag1"] = row["FSI1_lag1"]

x_s = np.array([row_s[c] for c in feature_cols]).reshape(1,-1)
pred_ridge, pred_rf, pred_cat = ridge.predict(x_s)[0], rf.predict(x_s)[0], cat.predict(x_s)[0]
exp_yield_s = ridge_exp.predict(pd.DataFrame([row_s[f] for f in features_for_expected]).T.values.reshape(1,-1))[0]
fri_scenario = pred_cat / (exp_yield_s + 1e-6)
risk_scenario = risk_from_fri(fri_scenario)

# ---------------------------
# Top metrics
# ---------------------------
col1,col2,col3,col4 = st.columns([1.8,1.2,1.2,1.2])
with col1:
    st.markdown(f"<div class='card'><h3>Division: {sel_div} — Year: {sel_year}</h3>"
                f"<div class='small-muted'>Scenario: precip {precip_mult}%, soil +{soil_add}, temp +{temp_add}°C, wind {wind_mult}%</div></div>",
                unsafe_allow_html=True)
with col2: st.metric("Predicted Yield (CatBoost)", f"{pred_cat:.2f}")
with col3: st.metric("Expected Yield (baseline)", f"{exp_yield_s:.2f}")
with col4:
    cls_map = {"Low":"risk-low","Medium":"risk-med","High":"risk-high"}
    cls = cls_map.get(risk_scenario,"risk-high")
    st.markdown(f"<div class='card'><div class='{cls}'>FRI: {fri_scenario:.3f} — {risk_scenario} Risk</div></div>", unsafe_allow_html=True)

# ---------------------------
# Visualizations
# ---------------------------
st.markdown("## Visualizations")
left, right = st.columns([2,1])
with left:
    df_div = df[df["Division"]==sel_div].sort_values("year")
    fig = px.line(df_div,x="year",y=[TARGET,"FRI"],markers=True,labels={"value":"Value","variable":"Metric"})
    st.plotly_chart(fig,use_container_width=True)

    latest = df.groupby("Division").apply(lambda g: g.loc[g["year"].idxmax()])[["Division","FRI"]].reset_index(drop=True)
    fig2 = px.bar(latest.sort_values("FRI"),x="Division",y="FRI",title="Latest FRI by Division")
    st.plotly_chart(fig2,use_container_width=True)

with right:
    st.subheader("Model comparison (test set)")
    st.table(eval_df)

    st.subheader("Top SHAP features (global)")
    try:
        shap_vals_sample = shap_vals[:min(100, shap_vals.shape[0])]
        fig_shap = shap.plots.beeswarm(shap_vals_sample, show=False)
        st.pyplot(bbox_inches='tight', dpi=150)
    except:
        fi = pd.DataFrame({'feature': feature_cols, 'importance': cat.get_feature_importance()})
        fig_bar = px.bar(fi.sort_values("importance",ascending=False),x="importance",y="feature",orientation='h')
        st.plotly_chart(fig_bar,use_container_width=True)

# ---------------------------
# Local SHAP
# ---------------------------
st.markdown("### Local explanation (scenario)")
with st.spinner("Computing local SHAP explanation..."):
    shap_values_point = explainer.shap_values(pd.DataFrame(x_s, columns=feature_cols))
    try:
        fig, ax = plt.subplots(figsize=(10,4))
        shap.plots.force(explainer.expected_value, shap_values_point[0], pd.DataFrame(x_s, columns=feature_cols), matplotlib=True, ax=ax)
        st.pyplot(fig)
    except:
        contrib = pd.Series(shap_values_point[0], index=feature_cols).abs().sort_values(ascending=False)[:10]
        fig_bar = px.bar(x=contrib.values,y=contrib.index,orientation='h',labels={'x':'|SHAP value|','y':'Feature'},title="Top local SHAP contributions")
        st.plotly_chart(fig_bar,use_container_width=True)
st.success("Local SHAP ready.")

# ---------------------------
# Report download
# ---------------------------
st.markdown("## Download report")
report_df = pd.DataFrame([{
    "Division": sel_div, "Year": sel_year,
    "Pred_CatBoost": pred_cat, "Pred_RF": pred_rf, "Pred_Ridge": pred_ridge,
    "Expected_Yield": exp_yield_s, "FRI_scenario": fri_scenario,
    "Risk": risk_scenario, "generated_at": datetime.utcnow().isoformat()
}])
csv_report = report_df.to_csv(index=False)
b64 = base64.b64encode(csv_report.encode()).decode()
download_link = f'<a href="data:file/csv;base64,{b64}" download="FRI_report_{sel_div}_{sel_year}.csv">📥 Download CSV report</a>'
st.markdown(download_link, unsafe_allow_html=True)

st.markdown("---")
st.caption("Note: This dashboard runs models on provided dataset.")
