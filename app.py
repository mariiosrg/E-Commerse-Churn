import joblib
import streamlit as st
import altair as alt
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the saved model, StandardScaler, and LabelEncoder
#X=scaled_df_encoded_smote[["WarehouseToHome","HourSpendOnApp","PreferedOrderCat","PreferredLoginDevice","PreferredPaymentMode","SatisfactionScore","Complain","CouponUsed","OrderCount","DaySinceLastOrder","CashbackAmount"]]
# y=scaled_df_encoded_smote['Churn']

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="	:iphone:",
    layout="wide",
    initial_sidebar_state="expanded")

## load dataset
path = ('E Commerce Dataset.xlsx')
df = pd.read_excel(path, sheet_name='E Comm')
X=["WarehouseToHome","HourSpendOnApp","SatisfactionScore","CouponUsed","OrderCount","DaySinceLastOrder","CashbackAmount"]

loaded_model = joblib.load('SVM_model.sav')
scaler = joblib.load('features_scaler.sav')
encoder = joblib.load('label_encoder.sav')

st.title("Customer Churn Prediction")
st.image("./header.png")
col1, col2, col3, col4,col5, col6, col7  = st.columns(7)

WarehouseToHome = st.number_input("WarehouseToHome(0-999)", 0, 999)

HourSpendOnApp = st.number_input("HourSpendOnApp (0-24) ", 0, 24)

SatisfactionScore =  st.number_input("SatisfactionScore (0-5) ", 0, 5)

CouponUsed = st.number_input("CouponUsed (0-20)", 0, 20)

OrderCount = st.number_input("OrderCount (1-20)", 1, 20)

DaySinceLastOrder = st.number_input("DaySinceLastOrder (0-50)", 0, 50)

CashbackAmount = st.number_input("CashbackAmount (0-99999):", 0, 99999)

# new_data = [[WarehouseToHome,HourSpendOnApp,PreferedOrderCat,PreferredLoginDevice,PreferredPaymentMode,SatisfactionScore,Complain,CouponUsed,OrderCount,DaySinceLastOrder,CashbackAmount]]
# new_data_scaled = scaler.transform(new_data)

# Data awal
new_data = [[WarehouseToHome, HourSpendOnApp, SatisfactionScore, CouponUsed, OrderCount, DaySinceLastOrder, CashbackAmount]]
encoded_df = pd.DataFrame(new_data, columns=X)

# Scale the features using the loaded StandardScaler
new_data_scaled = scaler.transform(encoded_df)

# Make predictions using the Random Forest model
predictions = loaded_model.predict(new_data_scaled)
# Decode the predicted target variable using the loaded LabelEncoder
if predictions == 1:
    prediction = "Churn"
else:
    prediction ="No Churn"

st.write("""
## Prediction
The predicted customer churn or no churn is:
""")
st.write(prediction)







