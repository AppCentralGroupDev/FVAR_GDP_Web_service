import streamlit as st 
import requests
import os
import math
import mlflow

import pandas as pd
import numpy as np


# from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode


# GDP Model 

st.title("GDP Model")

st.subheader(""" Forecast GDP """)


#Side bar
st.sidebar.write("Change your Exogenous values ")
TypeOfInputs = st.sidebar.selectbox("How Would You like to Distribute the Input",('Growth rate' , 'Custom Input'))


#os variables

os.environ["AWS_ACCESS_KEY_ID"] = "minio"	
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"	
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://host.docker.internal:9000"




base_url = 'http://host.docker.internal:8086/'
run_id = '0a22ff73acdb464f9158623a9494d3f7'
remote_server_uri = "http://mlflow_server:5000"

headers = {
    'accept': 'application/json',
    # Already added when you pass json= but not when you pass data=
    # 'Content-Type': 'application/json',
}

  # set to your server URI
mlflow.set_tracking_uri(remote_server_uri)

# growthrate_Df data analysis and cleaning 
growthrate_dates = pd.date_range(start='2000-03-31', end='2022-12-31',freq='q')
growthrate_Df = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/growth_rate.xlsx" ,dst_path=os.getcwd())
growthrate_Df = pd.read_excel(growthrate_Df)

# growthrate_combine = pd.concat([growthrate_dates,growthrate_Df], axis=1)
growthrate_Df.index = growthrate_dates


# test data 
if TypeOfInputs == 'Growth rate':
    growthrate = st.sidebar.slider("Growth Rate",0,100,10)
    COP = st.sidebar.slider("Crude Oil Price", 0, 120, 50)
    GXP = st.sidebar.slider("Govt Expenditure", 1000000, 99999999, 4024610)
    MPMIS = st.sidebar.slider("Manufacturing Index", 0, 100, 51)

    COP2 = COP * (1 + (growthrate/100))
    COP3 = COP2 * (1 + (growthrate/100))
    COP4 = COP3 * (1 + (growthrate/100))

    GXP2 = GXP * (1 + (growthrate/100))
    GXP3 = GXP2 * (1 + (growthrate/100))
    GXP4 = GXP3 * (1 + (growthrate/100))


    MPMIS2 = MPMIS * (1 + (growthrate/100))
    MPMIS3 = MPMIS2 * (1 + (growthrate/100))
    MPMIS4 = MPMIS3 * (1 + (growthrate/100))


    
    pass
else:
    
    

    COP = st.sidebar.number_input('Insert Crude Oil Price for Quarter 1',0, 120, 50)
    COP2 = st.sidebar.number_input('Insert Crude Oil Price for Quarter 2',0, 120, 50)
    COP3 = st.sidebar.number_input('Insert Crude Oil Price for Quarter 3',0, 120, 50)
    COP4 = st.sidebar.number_input('Insert Crude Oil Price for Quarter 4',0, 120, 50)

    GXP = st.sidebar.number_input('Insert Govt Expenditure  for Quarter 1', 1000000, 99999999, 4024610)
    GXP2 = st.sidebar.number_input('Insert Govt Expenditure for Quarter 2', 1000000, 99999999, 4024610)
    GXP3 = st.sidebar.number_input('Insert Govt Expenditure for Quarter 3', 1000000, 99999999, 4024610)
    GXP4 = st.sidebar.number_input('Insert Govt Expenditure for Quarter 4', 1000000, 99999999, 4024610)

    MPMIS = st.sidebar.number_input('Insert Manufacturing Index for Quarter 1', 0, 100, 51)
    MPMIS2 = st.sidebar.number_input('Insert Manufacturing Index for Quarter 2', 0, 100, 51)
    MPMIS3 = st.sidebar.number_input('Insert Manufacturing Index for Quarter 3', 0, 100, 51)
    MPMIS4 = st.sidebar.number_input('Insert Manufacturing Index for Quarter 4', 0, 100, 51)

    pass





json_data = [
    {
        'date': '2022-03-31',
        'COP': COP,
        'GXP': GXP,
        'MPMIS': MPMIS,
    },
    {
        'date': '2022-06-30',
        'COP': COP2,
        'GXP': GXP2,
        'MPMIS': MPMIS2,
    },
    {
        'date': '2022-09-30',
        'COP': COP3,
        'GXP': GXP3,
        'MPMIS': MPMIS3,
    },
    {
        'date': '2022-12-31',
        'COP': COP4,
        'GXP': GXP4,
        'MPMIS': MPMIS4,
    },
]


# st.write("Input Data ")
# json_data

input_df = pd.json_normalize(json_data).set_index('date')
input_df = np.log(input_df)
# input_df
new_input_dct = pd.DataFrame.to_dict(input_df)
# new_input_dct
response = requests.post(f'{base_url}predict/json/', headers=headers, json=json_data)
p_json = response.json()

print(p_json)



# st.write("Output Data")
# p_json


# df = pd.DataFrame.from_dict(p_json , orient="index" , columns=["dlRY"]).reset_index()
# #converting index from string to date 
# df.columns = ["date", "dlRY"]
# df['date'] = pd.to_datetime(df['date'])
# df.set_index(['date'] , inplace=True)

df = pd.DataFrame.from_dict(p_json , orient='index' , columns=["dlRY"])
df.index = pd.to_datetime(df.index)

df_percentage_change = df.copy().pct_change().fillna(0)



#Layout 

col1 , col2 , col3 , col4 = st.columns(4)

col1.metric(label='Quarter 1',value= df['dlRY'][0].round(3), delta=df_percentage_change['dlRY'][0].round(2))
col2.metric(label='Quarter 2',value= df['dlRY'][1].round(3), delta=df_percentage_change['dlRY'][1].round(2))
col3.metric(label='Quarter 3',value= df['dlRY'][2].round(3), delta=df_percentage_change['dlRY'][2].round(2))
col4.metric(label='Quarter 4',value= df['dlRY'][3].round(3), delta=df_percentage_change['dlRY'][3].round(2))



# st.dataframe(df)
#Graphs for Plotting Predictions

st.line_chart(df)
st.area_chart(df)

st.bar_chart(df)




#Growth rate graphs and Analysis 

gg = growthrate_Df['RY']
gg.drop(gg.tail(4).index,inplace=True)
frames = [gg, df['dlRY']]




st.subheader('Full Forecast')
forecastfull = pd.concat(frames)
st.line_chart(forecastfull)


st.subheader('Percentage Change')

percent_change_forecastfull = forecastfull.copy().pct_change().fillna(0)

st.line_chart(percent_change_forecastfull)






