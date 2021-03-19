
from fastapi import FastAPI

import joblib
import pandas as pd



app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}

# Running a few tests
# @app.get("/test")
# def return_numbers(first_number: int, second_number: int):
#     return {"number_1": first_number, "number_2": second_number}


# Implement a /predict endpoint
@app.get("/predict")
def run_prediction(
    acousticness: float,
    danceability: float,
    duration_ms: int,
    energy: float,
    explicit: int,
    id: str,
    instrumentalness: float,
    key: int,
    liveness: float,
    loudness: float,
    mode: int,
    name: str,
    release_date: str,
    speechiness: float,
    tempo: float,
    valence: float,
    artist: str
):
    query = {}

    for k, v in locals().items():
        query[k] = [v]

    one_row_X = pd.DataFrame.from_dict(query)
    model = joblib.load('model.joblib')

    y_pred = model.predict(one_row_X)

    return {
        "artist": artist,
        "name": name,
        "popularity": int(y_pred[0])
    }


