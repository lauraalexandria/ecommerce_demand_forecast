from io import BytesIO
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile

import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
client = mlflow.MlflowClient()
print("aqui")

app = FastAPI()

# # Load current and reference data
# current_data = pd.read_csv("./data/processed/x_val.csv").drop(
#     "order_purchase_date", axis=1
# )
# features = current_data.columns

# # Load model
# model = CatBoost()
# model.load_model("final_model/model/model.cb")
# model.set_feature_names(features)

# Load model
model = mlflow.pyfunc.load_model("models:/ecommerce_forecast/1")
print("ou aqui?")


@app.post("/predict-csv")
async def predict_csv(file: Optional[UploadFile] = None):
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Only CSV files are accepted"
        )

    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents)).drop(
            "order_purchase_date", axis=1
        )
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Processing Error: {str(e)}"
        ) from e
