#Importing Necessary Modules
from prophet.serialize import model_from_json
#import pickle
import streamlit as st
import pandas as pd

#Loading the Model from json file
with open('/Users/nilaysaraf/Downloads/bitcoin_prophet.json', 'r') as fin:
    prophet_model = model_from_json(fin.read())

#Preparing Dataframe for Prediction
f_df = prophet_model.make_future_dataframe(periods=365)

#Making Predictions
y_preds = prophet_model.predict(f_df)

#Dataframe with Required Columns
view_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
view_df = y_preds[view_cols]
view_df.columns = ["Date", "Predicted Forecast", "Lower Bound", "Upper Bound"]

#Frontend Using Streamlit
st.title("Bitcoin Price Prediction")
date = st.date_input("Select Date: ")
date = date.strftime("%Y-%m-%d")
st.divider()

results = view_df[view_df["Date"] == date]
ds = [i for i in range(1,len(view_df)+1)]
new_df = pd.DataFrame({"Day": ds, "Predicted": view_df["Predicted Forecast"]})

st.header("For the Selected Date:")
st.write("Predicted Price : ", results["Predicted Forecast"].values[0])
st.write("Lower Bound     : ", results["Lower Bound"].values[0])
st.write("Upper Bound     : ", results["Upper Bound"].values[0])

st.divider()
st.line_chart(new_df)
