"""WangchanBERTa classifier. Migrated from sentence-classification/train_pytorch.py."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from src.config import MAX_LEN_BERT, WANGCHANBERTA_MODEL_NAME


class WangchanBERTaDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label_map: dict):
        self.labels = [label_map[label] for label in df["label"]]
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=MAX_LEN_BERT,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["text"]
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], np.array(self.labels[idx])


class WangchanBERTaClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = WANGCHANBERTA_MODEL_NAME,
        dropout: float = 0.5,
        num_classes: int = 4,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        return self.relu(self.linear(self.dropout(pooled_output)))


def load_tokenizer(model_name: str = WANGCHANBERTA_MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)
