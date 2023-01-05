import os

import numpy as np
import uuid

import json
import mlflow
import pandas as pd
import io
from fastapi import FastAPI, File, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse

import boto3
import requests
from pymongo import MongoClient
from mlflow.tracking import MlflowClient

from pydantic import BaseModel
from datetime import datetime, time, timedelta, date
from typing import List, Union

import logging
boto3.set_stream_logger('boto3.resources', logging.NOTSET)

EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://127.0.0.1:5000")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")

class exoRow(BaseModel):

    date: Union[ datetime , date ]
    COP: float
    GXP: float
    MPMIS: float

# Create FastAPI instance
app = FastAPI(title="CBN GDP Developer Guide")
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database("prediction_service")
collection = db.get_collection("data")


def save_to_db(record):
    rec = record.copy()
    collection.insert_many(rec)


def send_to_evidently_service(record):
    #rec = record.copy()
    # requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/gdp", data=record)
    requests.post(f"{EVIDENTLY_SERVICE_ADDRESS}/iterate/gdp", files={"file": record})





os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://host.docker.internal:9000"



# remote_server_uri = "http://localhost:5001"# set to your server URI
remote_server_uri = "http://mlflow_server:5000"# set to your server URI

mlflow.set_tracking_uri(remote_server_uri)
# model_name = "cbnGDP"
model_version = os.getenv("modelVersion", "")
run_ID = os.getenv("run_ID", "")
model_name = os.getenv("modelName", "")


model = mlflow.statsmodels.load_model(model_uri=f"models:/{model_name}/{model_version}")
endog = pd.read_csv("endog.csv")
endog = endog.set_index("date")

df = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_ID}/favardata1105.xlsx" ,dst_path=os.getcwd())
endog = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_ID}/endog.csv" ,dst_path=os.getcwd())

endog = pd.read_csv(endog)
endog = endog.set_index("date")

df = pd.read_excel(df)
df = df.set_index('date')
dflog = np.log(df[[ 'ABCPI', 'ARY', 'ASI', 'BLAG', 'BLMF', 'BLOG', 'BLPS', 'BLSM', 'BLTL', 'BLUS', 'BLXP', 'C1CPI', 'C2CPI',
                    'CCPI', 'CCPS', 'CFCPI', 'CGRY', 'COP', 'CPD', 'CPS', 'COS', 'CRY', 'ECPI', 'ER', 'EUR', 'EXR', 'FCPI',
                    'FHCPI', 'FNCPI', 'FRY', 'GBP', 'GRV', 'GXP', 'HHCPI', 'HRY', 'HWCPI', 'IEP', 'IIP', 'IMAP', 'IMIP',
                    'IMP', 'IRY', 'M1', 'M2', 'MCPI', 'MRY', 'NDC', 'NFA', 'NORY', 'PRY', 'QM', 'RCCPI', 'RHCPI', 'RINV',
                    'RPC', 'RPDI', 'RR', 'RRY', 'RUCPI', 'RY', 'SD', 'SMRY', 'SRY', 'TCPI', 'TD', 'TRY', 'URCPI', 'URY',
                    'USD', 'EXP', 'MPMIS' ]])
@app.post("/predict")
async def predict(request: Request, file: bytes = File(...)):
    print('[+] Initiate Prediction')
    file_obj = io.BytesIO(file)
    file_df = pd.read_csv(file_obj)
    test_df = file_df.set_index("date")
    print(test_df)

    #request_json = await request.json()

    preds = model.forecast(endog.values[-1:], steps=4, exog_future=test_df)
    preds = pd.DataFrame(data = preds, columns = ['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'dlRY'], index=test_df.index)
    pred = preds[['dlRY']]

    test_df['dlRY'] = preds['dlRY']

    values = test_df.to_dict('records')
    json_values = test_df.to_json()
    print(json_values)





    save_to_db(values)
    # send_to_evidently_service(json_values)
    send_to_evidently_service(file)
    json_compatible_item_data = jsonable_encoder(pred)
    return JSONResponse(content=json_compatible_item_data)


@app.post("/predict/json/")
async def predict_json(request: Request, items: List[exoRow]):
    items_dict = jsonable_encoder(items)

    test_df = pd.DataFrame.from_dict(items_dict).set_index('date')

    ########### additional Testing Done#################
    exog = dflog[['COP', 'MPMIS', 'GXP']]
    exog.drop(exog.tail(4).index, inplace=True)
    # print(exog)

    # test_df = pd.DataFrame(test_df,index= df[-4:].index)
    test_df = np.log(test_df)
    # print(test_df)

    frames = [exog, test_df]
    exogOI = pd.concat(frames)

    shiftofgxp = exogOI['GXP'] - exogOI['GXP'].shift(4)
    print(shiftofgxp)
    exogOI = exogOI[['COP', 'MPMIS']]
    exogOI = pd.merge(exogOI, shiftofgxp, on=['date'])
    # print(exogOI)
    exogModel = exogOI.loc['2017-03-31':'2021-12-31', ['COP', 'GXP', 'MPMIS']]
    exogForecast = exogOI.loc['2022-03-31':'2022-12-31', ['COP', 'GXP', 'MPMIS']]
    # print(exogForecast)
    # frames = [exog, input_exog]
    preds = model.forecast(endog.values[-1:], steps=4, exog_future=exogForecast)
    preds = pd.DataFrame(data=preds, columns=['pc1', 'pc2', 'pc3', 'pc4', 'pc5', 'dlRY'], index=exogForecast.index)
    # pred = preds[['dlRY']]
    pred_s = preds['dlRY']
    test_df['dlRY'] = preds['dlRY']
    # creating a clone of the exgo
    exogForecast_temp = exogForecast.copy()

    # renaming columns
    exogForecast_temp.rename(columns={
        'COP':'COP50',
        'GXP': 'GXP1',
        'MPMIS': 'MPMIS1'
    }, inplace = True)
    print(exogForecast_temp)

    #saving tmp file
    tmp_uuid = uuid.uuid1()
    tmp_path = f'temp_${str(tmp_uuid)}.csv'
    if not os.path.exists(tmp_path):
        exogForecast_temp.to_csv(tmp_path)
        pass

    exogForecast_temp['dlRY'] = preds['dlRY']

    print(exogForecast_temp)

    values = exogForecast_temp.to_dict('records')
    print(values)

    json_values = exogForecast_temp.to_json()
    print(json_values)

    # add mertics

    #create temp file in directory


    file = open(tmp_path)





    # add mertics
    save_to_db(values)
    send_to_evidently_service(file)

    #removing tmp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)




    #remove temp_file in directory

    json_compatible_item_data = jsonable_encoder(pred_s)

    # json_compatible_item_data = jsonable_encoder(preds)
    return JSONResponse(content=json_compatible_item_data)

@app.get("/")
async def main():
    content = """
    <body>
    <form action="/predict/" enctype="multipart/form-data" method="post">
    <input name="file" type="file" multiple>
    <input type="submit">
    </form>
    </body>
     """
    return HTMLResponse(content=content)

