"""FastAPI serving. Migrated from sentiment-classification-api/main.py.

Supports BiLSTM (default, legacy weights) — WangchanBERTa serving TODO.
Run: uvicorn src.api.main:app --host 0.0.0.0 --port 8080
"""
from enum import Enum

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.models.bilstm import BiLSTMSentimentModel

model = BiLSTMSentimentModel()

app = FastAPI(title="Wisesight sentiment classification")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


class Sentiment(str, Enum):
    q = "q"
    neg = "neg"
    neu = "neu"
    pos = "pos"


@app.get("/")
def read_root():
    return {
        "API": "Wisesight sentiment classification",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.post("/predict")
def predict_sentiment(body: TextInput):
    sentiment, prob = model.predict(body.text)
    return {"sentiment": sentiment, "probability": prob}


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
