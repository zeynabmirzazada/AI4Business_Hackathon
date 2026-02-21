import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration & Title
st.set_page_config(page_title="AI-VentureScore", layout="wide")
st.title("🔮 AI-VentureScore: Startup Risk Predictor")
st.markdown("Adjust the metrics in the sidebar to predict the likelihood of this startup succeeding or failing.")

# 2. Train the Model 
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv('startup_risk_data.csv')
    X = df.drop(columns=['Startup_Name', 'Company_Status'])
    y = df['Company_Status']
    
    categorical_columns = ['Industry_Sector']
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)
    
    return model, X_encoded.columns, X

rf_model, expected_features, original_X = load_and_train_model()

# 3. Sidebar: User Inputs
st.sidebar.header("📊 Startup Metrics")

cash_runway = st.sidebar.slider("Cash Runway (Months)", min_value=0, max_value=60, value=12)
monthly_churn = st.sidebar.slider("Monthly Churn Rate (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
yoy_growth = st.sidebar.slider("YoY Growth (%)", min_value=-100, max_value=500, value=40)
mrr = st.sidebar.number_input("Monthly Recurring Revenue ($)", min_value=0, value=50000, step=10000)
burn_rate = st.sidebar.number_input("Estimated Monthly Burn Rate ($)", min_value=0, value=60000, step=10000)
team_size = st.sidebar.number_input("Current Team Size", min_value=1, value=15)
industry = st.sidebar.selectbox("Industry Sector", options=original_X['Industry_Sector'].unique())

# 4. Process Inputs & Predict
# FIXED: Lowered the "hidden" baseline metrics so the sliders actually matter!
user_input = {
    'YoY_Growth_Pct': yoy_growth,
    'Investment_Readiness_Score': 50,     # Lowered from 85
    'Number_of_Founders': 2,
    'Founder_Previous_Exits': 0,          # Lowered from 1
    'Current_Team_Size': team_size,
    'Employee_MoM_Growth': 1.0,           # Lowered from 5.0
    'Total_Funding_USD': 1500000,         # Lowered from $15M to $1.5M
    'Funding_Rounds_Count': 1,
    'Months_Since_Last_Funding': 12,      # Increased time since last funding
    'Estimated_Monthly_Burn_Rate': burn_rate,
    'Cash_Runway_Months': cash_runway,
    'Monthly_Recurring_Revenue_MRR': mrr,
    'Customer_Acquisition_Cost_CAC': 300, # Increased CAC
    'Customer_Lifetime_Value_LTV': 600,   # Lowered LTV (Now a mediocre 2:1 ratio)
    'Monthly_Churn_Rate': monthly_churn,
    'Company_Age_Months': 24,
    'Industry_Sector': industry,
    'Web_Traffic_Growth_MoM': 2.0,        # Lowered from 15.0
    'Media_Mentions_Count': 5
}

# Convert to DataFrame and encode
input_df = pd.DataFrame([user_input])
input_encoded = pd.get_dummies(input_df, columns=['Industry_Sector'])
input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

# Make the prediction
prediction = rf_model.predict(input_encoded)
probabilities = rf_model.predict_proba(input_encoded)[0]

# 5. Display the Results
st.divider()
st.subheader("Model Verdict")

if prediction[0] == 0:
    st.success(f"✅ **SUCCESS** (Low Risk / High Potential)")
    st.write(f"**AI Confidence Level:** {probabilities[0] * 100:.1f}%")
else:
    st.error(f"❌ **FAILURE** (High Risk of Closure)")
    st.write(f"**AI Confidence Level:** {probabilities[1] * 100:.1f}%")

st.divider()
st.subheader("Key Indicator Check")
col1, col2, col3 = st.columns(3)
col1.metric("Runway Health", f"{cash_runway} Months", "Critical" if cash_runway < 6 else "Stable", delta_color="inverse" if cash_runway < 6 else "normal")
col2.metric("Churn Risk", f"{monthly_churn}%", "High" if monthly_churn > 10 else "Low", delta_color="inverse" if monthly_churn > 10 else "normal")
col3.metric("Growth Velocity", f"{yoy_growth}%", "Warning" if yoy_growth < 0 else "Good", delta_color="inverse" if yoy_growth < 0 else "normal")