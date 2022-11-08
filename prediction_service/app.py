import os

import numpy as np

import json
import mlflow
import pandas as pd
import io
from fastapi import FastAPI, File, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, HTMLResponse
import requests
from pymongo import MongoClient
from mlflow.tracking import MlflowClient


EVIDENTLY_SERVICE_ADDRESS = os.getenv("EVIDENTLY_SERVICE", "http://127.0.0.1:5000")
MONGODB_ADDRESS = os.getenv("MONGODB_ADDRESS", "mongodb://127.0.0.1:27017")



# Create FastAPI instance
app = FastAPI()
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
os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://172.18.0.5:9000"



# remote_server_uri = "http://localhost:5001"# set to your server URI
remote_server_uri = "http://mlflow_server:5000"# set to your server URI

mlflow.set_tracking_uri(remote_server_uri)
model_name = "cbnGDP"
model_version = 1

model = mlflow.statsmodels.load_model(model_uri=f"models:/{model_name}/{model_version}")
endog = pd.read_csv("endog.csv")
endog = endog.set_index("date")


@app.post("/predict")
async def predict(request: Request, file: bytes = File(...) ):
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

