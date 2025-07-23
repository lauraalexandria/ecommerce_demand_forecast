# from pydantic import BaseModel
# from typing import List

import pandas as pd
from catboost import CatBoost
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse  # , HTTPException

# Load current and reference data
current_data = pd.read_csv("./data/processed/x_val.csv").drop(
    "order_purchase_date", axis=1
)
features = current_data.columns

# Load model
model = CatBoost()
model.load_model("final_model/model/model.cb")
model.set_feature_names(features)

app = FastAPI()

feature_types = {
    col: (
        "number"
        if pd.api.types.is_numeric_dtype(current_data[col])
        else "text"
    )
    for col in current_data.columns
}


# Root for human interface - HTML
@app.get("/", response_class=HTMLResponse)
async def home():
    # Generate inputs
    form_fields = ""
    for feature, input_type in feature_types.items():
        if input_type == "number":
            form_fields += f"""
            <label for="{feature}">{feature} (num):</label>
            <input type="number" name="{feature}" step="0.01" required><br>
            """
        else:
            form_fields += f"""
            <label for="{feature}">{feature} (str):</label>
            <input type="text" name="{feature}" required><br>
            """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body>
        <h2>Predict Form</h2>
        <form action="/predict" method="post">
            {form_fields}
            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    """
    return html_content


@app.post("/predict")
async def predict(request: Request):
    form_data = await request.form()
    return {"received_data": dict(form_data)}


# @app.post("/predict")
# async def predict(request: Request):
#     form_data = await request.form()
#     try:
#         input_data = {
#             "feature1": float(form_data["feature1"]),
#             "feature2": float(form_data["feature2"]),
#         }
#         return {"prediction": "sucesso", "data": input_data}
#     except Exception as e:
#         raise HTTPException(
#             status_code=400, detail=f"Erro nos dados: {str(e)}"
#         )
