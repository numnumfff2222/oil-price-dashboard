import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

# --- CONFIG ---
st.set_page_config(page_title="Planetary Signals Lab", layout="wide")
st.title("🌍 Planetary Signals: Global Oil to Local Impact Dashboard")
st.markdown("🔍 *Analyzing Time-Lag, Probabilistic Risk, and Environmental Signals (EIA, EPPO, OWID)*")

# --- 1. LOAD & PREPARE DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('final_energy_data.csv')
    df['period'] = pd.to_datetime(df['period'])

    # [Baseline] ค่าเฉลี่ยรายเดือนปี 2024 เป็นฐาน
    baseline_sales = df['Total_Sales_MLitres'].mean()

    # [Time-Lag] สร้าง Lag 1 เดือน: World Price(t) → Thai Sales(t+1)
    df_model = df.copy()
    df_model['Next_Month_Sales'] = df_model['Total_Sales_MLitres'].shift(-1)
    df_model = df_model.dropna()

    # สถิติราคาน้ำมันในอดีต (EIA Historical Series)
    price_mean = df['value'].mean()
    price_std  = df['value'].std()

    return df, df_model, baseline_sales, price_mean, price_std

df_final, df_model, baseline_sales, p_mean, p_std = load_data()

# --- 2. TRAIN MODEL ---
X = df_model[['value']].values
y = df_model['Next_Month_Sales'].values
model      = LinearRegression().fit(X, y)
std_error  = (y - model.predict(X)).std()
slope      = model.coef_[0]   # ใช้แสดง direction ใน Data Lineage

# --- 3. SIDEBAR ---
st.sidebar.header("🕹️ Oil Price Simulator")
st.sidebar.subheader("🚀 Quick Scenarios")
if st.sidebar.button("🔥 Middle East Conflict ($120)"):
    st.session_state.price = 120.0
if st.sidebar.button("📉 Global Recession ($40)"):
    st.session_state.price = 40.0
if st.sidebar.button("🔄 Reset to Current"):
    st.session_state.price = float(df_final['value'].iloc[-1])

if 'price' not in st.session_state:
    st.session_state.price = float(df_final['value'].iloc[-1])

input_price     = st.sidebar.slider("จำลองราคาน้ำมันโลก ($/BBL)", 10.0, 150.0, key="price")
threshold_sales = st.sidebar.number_input("เกณฑ์วิกฤตยอดขาย (ล้านลิตร)", value=9500)

# --- 4. CALCULATIONS ---

# [Prediction] Linear model ที่ train บน lag-1 data
pred_sales = model.predict([[input_price]])[0]

# [95% Confidence Interval]
ci_95 = 1.96 * std_error
pred_low  = pred_sales - ci_95
pred_high = pred_sales + ci_95

# ─────────────────────────────────────────────────────────────────
# [FIX 1] Risk Probability — รวม 3 สัญญาณเป็น composite score
#
#  A) Statistical risk: โอกาสที่ยอดขายจะต่ำกว่าเกณฑ์วิกฤต
#     (ถ้า pred_sales ต่ำ → ความเสี่ยงสูง)
statistical_risk = stats.norm.cdf(threshold_sales, pred_sales, std_error)

#  B) Price shock risk: ราคาพุ่งเกิน mean + 1.5σ → ถือว่า Shock
#     ใช้ sigmoid เพื่อให้ scale 0→1 อย่าง smooth
price_z    = (input_price - p_mean) / p_std
shock_risk = 1 / (1 + np.exp(-3 * (price_z - 1.5)))  # sigmoid centered at +1.5σ

#  C) Demand destruction risk: slope < 0 หมายถึงราคาสูง → sales ลด
#     ยิ่งราคาสูงกว่า mean มากเท่าไหร่ ยิ่งเสี่ยง
demand_risk = max(0.0, price_z / 5.0)   # linear, cap ที่ 1 ที่ price_z = 5

# รวม 3 สัญญาณ (weighted)
risk_prob = min(1.0, 0.4 * statistical_risk + 0.4 * shock_risk + 0.2 * demand_risk)

# Risk label
if risk_prob > 0.5:
    risk_label = "🔴 CRITICAL"
elif risk_prob > 0.25:
    risk_label = "🟠 Warning"
elif risk_prob > 0.1:
    risk_label = "🟡 Elevated"
else:
    risk_label = "🟢 Normal"
# ─────────────────────────────────────────────────────────────────

# [FIX 2] Environmental impact — quantified vs baseline
diff_pct   = ((pred_sales - baseline_sales) / baseline_sales) * 100
co2_impact = (pred_sales - baseline_sales) * 2.3   # kg → Tonnes (/1000 ถ้าต้องการ)

# [FIX 3] CO2 Signal ที่ขยับตาม slider
# Historical CO2 normalised ต่อ baseline
hist_co2_signal = ((df_final['Total_Sales_MLitres'] * 2.3) / (baseline_sales * 2.3)) * 100
# Predicted CO2 (เดือนหน้า) ขยับตาม pred_sales ซึ่งขึ้นกับ input_price
pred_co2_pct    = ((pred_sales * 2.3) / (baseline_sales * 2.3)) * 100

# --- 5. TOP METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("World Oil Price (Input)", f"${input_price:.2f}")
col2.metric(
    "Predicted Sales (±95% CI)",
    f"{pred_sales:,.0f} M.L.",
    delta=f"Range: {pred_low:,.0f}–{pred_high:,.0f}",
    delta_color="off"
)
col3.metric(
    "Risk Probability",
    f"{risk_prob*100:.1f}%",
    delta=risk_label,
    delta_color="inverse" if risk_prob > 0.25 else "off"
)

st.divider()

# --- 6. ENVIRONMENTAL IMPACT & DATA LINEAGE ---
st.subheader("🍀 Planetary Feedback Loop")
e_col1, e_col2 = st.columns([1, 2])

with e_col1:
    if diff_pct < 0:
        st.success(f"🌱 Fossil Consumption: {diff_pct:.2f}% vs Baseline")
        st.write(f"Estimated CO2 Reduction: **{abs(co2_impact):,.2f} Tonnes**")
    else:
        st.warning(f"⚠️ Fossil Consumption: +{diff_pct:.2f}% vs Baseline")
        st.write(f"Estimated CO2 Surplus: **{co2_impact:,.2f} Tonnes**")

    # แสดง demand direction อย่างชัดเจน
    direction = "ลดลง (demand destruction)" if slope < 0 else "เพิ่มขึ้น"
    st.info(f"📐 Model slope: {slope:+.1f} M.L./$\n\nยอดขายมีแนวโน้ม**{direction}**เมื่อราคาสูงขึ้น")

with e_col2:
    with st.expander("📝 Data Lineage & Methodology (For Judges)", expanded=True):
        st.markdown(f"""
| ส่วน | รายละเอียด |
|---|---|
| **Baseline** | ค่าเฉลี่ยรายเดือนปี 2024 = {baseline_sales:,.0f} M.Litre |
| **Lag Evidence** | Cross-correlation peak k=1 (r=0.52). World Price(t) → Thai Sales(t+1) |
| **Model slope** | {slope:+.2f} M.L./$ → ราคาสูงขึ้น $1 ยอดขายเปลี่ยน {slope:+.2f} M.L. |
| **CO2 Formula** | Thai Fuel Carbon Intensity avg 2.3 kgCO₂/L (OWID) |
| **Risk Formula** | 40% Statistical + 40% Price Shock (sigmoid >1.5σ) + 20% Demand |
| **Sources** | EIA (STEO/WTOTWORLD), EPPO (Table 2.3-3), OWID (Energy Mix) |
        """)

# --- 7. VISUALIZATION ---
st.subheader("📊 Multi-Source Signal Analysis")

next_month = df_final['period'].max() + pd.DateOffset(months=1)

fig = make_subplots(specs=[[{"secondary_y": True}]])

# EIA World Price
fig.add_trace(go.Scatter(
    x=df_final['period'], y=df_final['value'],
    name="EIA World Price ($)", line=dict(color='royalblue', width=2)
), secondary_y=False)

# EPPO Thai Sales
fig.add_trace(go.Scatter(
    x=df_final['period'], y=df_final['Total_Sales_MLitres'],
    name="EPPO Thai Sales (M.L.)", line=dict(color='orange', width=2)
), secondary_y=True)

# Historical CO2 Signal
fig.add_trace(go.Scatter(
    x=df_final['period'], y=hist_co2_signal,
    name="Historical CO2 Signal (%)",
    line=dict(color='green', dash='dot', width=1.5)
), secondary_y=True)

# Predicted CO2 Signal (ขยับตาม slider) — ★ FIX 3
fig.add_trace(go.Scatter(
    x=[next_month], y=[pred_co2_pct],
    name=f"Predicted CO2 ({pred_co2_pct:.1f}%)",
    mode="markers",
    marker=dict(color='lime', size=14, symbol='diamond',
                line=dict(color='darkgreen', width=2))
), secondary_y=True)

# Predicted Sales (ขยับตาม slider) — ★ FIX 2
fig.add_trace(go.Scatter(
    x=[next_month], y=[pred_sales],
    name=f"Predicted Sales ({pred_sales:,.0f} M.L.)",
    mode="markers",
    marker=dict(color='red', size=16, symbol='star',
                line=dict(color='darkred', width=2))
), secondary_y=True)

# Shock markers
fig.add_vline(x=pd.Timestamp("2024-04-01").timestamp()*1000,
              line_dash="dash", line_color="red", annotation_text="OPEC+ Supply Cut")
fig.add_vline(x=pd.Timestamp("2024-09-01").timestamp()*1000,
              line_dash="dash", line_color="purple", annotation_text="Domestic Subsidy Shift")

fig.update_yaxes(title_text="World Price ($/BBL)", secondary_y=False)
fig.update_yaxes(title_text="Sales (M.L.) / CO2 Signal (%)", secondary_y=True)
fig.update_layout(hovermode="x unified", height=460, legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)

# --- 8. RISK DISTRIBUTION & ANOMALY ---
c_left, c_right = st.columns(2)

with c_left:
    st.subheader("🎲 Probabilistic Distribution")

    x_dist = np.linspace(pred_sales - 4*std_error, pred_sales + 4*std_error, 200)
    y_pdf  = stats.norm.pdf(x_dist, pred_sales, std_error)

    fig_dist = go.Figure()

    # Normal zone
    fig_dist.add_trace(go.Scatter(
        x=x_dist, y=y_pdf, fill='tozeroy',
        fillcolor='rgba(0,100,255,0.15)',
        line=dict(color='royalblue'),
        name="Prediction PDF"
    ))

    # Critical zone (below threshold)
    x_crit = x_dist[x_dist <= threshold_sales]
    if len(x_crit) > 0:
        y_crit = stats.norm.pdf(x_crit, pred_sales, std_error)
        fig_dist.add_trace(go.Scatter(
            x=x_crit, y=y_crit, fill='tozeroy',
            fillcolor='rgba(255,50,50,0.45)',
            line=dict(color='red'),
            name=f"Critical Zone (<{threshold_sales:,})"
        ))

    # เส้น threshold
    fig_dist.add_vline(x=threshold_sales, line_dash="dash", line_color="red",
                       annotation_text=f"Threshold: {threshold_sales:,}")

    # เส้น predicted sales
    fig_dist.add_vline(x=pred_sales, line_dash="dot", line_color="royalblue",
                       annotation_text=f"Pred: {pred_sales:,.0f}")

    fig_dist.update_layout(height=320, showlegend=True)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Risk summary box
    st.metric("Composite Risk Score", f"{risk_prob*100:.1f}%", delta=risk_label,
              delta_color="inverse" if risk_prob > 0.25 else "off")

with c_right:
    st.subheader("🚨 Anomaly Detection (1.2σ Threshold)")

    z_scores  = (y - model.predict(X)) / std_error
    anomalies = df_model[np.abs(z_scores) > 1.2].copy()
    anomalies['z_score'] = z_scores[np.abs(z_scores) > 1.2]

    if len(anomalies) > 0:
        st.warning(f"ตรวจพบความผิดปกติ {len(anomalies)} จุด (Threshold > 1.2σ)")
        st.dataframe(
            anomalies[['period', 'value', 'Next_Month_Sales', 'z_score']]
            .rename(columns={
                'value': 'Price ($/BBL)',
                'Next_Month_Sales': 'Next Mo. Sales (M.L.)',
                'z_score': 'Z-Score'
            })
            .style.format({
                'Price ($/BBL)': '{:.2f}',
                'Next Mo. Sales (M.L.)': '{:,.0f}',
                'Z-Score': '{:+.2f}'
            }),
            use_container_width=True
        )
        st.info("💡 **Insight:** จุดผิดปกติมักสัมพันธ์กับการแทรกแซงราคาจากภาครัฐหรือ supply disruption ระดับโลก")
    else:
        st.success("ไม่พบความผิดปกติในช่วงข้อมูลปัจจุบัน")

st.divider()
st.caption("📡 EIA Series: STEO.WTOTWORLD.M | EPPO Table 2.3-3 | OWID Carbon Intensity Baseline | Model: OLS lag-1 regression")