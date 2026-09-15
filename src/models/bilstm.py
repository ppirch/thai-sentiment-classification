"""BiLSTM model. Migrated from sentiment-classification-api/model.py."""
import pickle
from pathlib import Path

import numpy as np
import torch
from pythainlp import word_tokenize
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.config import BILSTM_ID2LABEL
from src.preprocess import preprocess


class BILSTM_Model(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 4)

    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        _out_pack, (ht, _ct) = self.lstm(x_pack)
        x = self.linear1(torch.cat((ht[-2, :, :], ht[-1, :, :]), dim=1))
        return self.linear2(x)


class BiLSTMSentimentModel:
    """Inference wrapper (was MySentimentModel)."""

    def __init__(
        self,
        vocab_path: str = "artifacts/vocab2index.pkl",
        id2label_path: str = "artifacts/bilstm_id2label.pkl",
        weights_path: str = "artifacts/stacked_bilstm_model.pth",
    ) -> None:
        with open(vocab_path, "rb") as f:
            self.vocab2index = pickle.load(f)
        if Path(id2label_path).exists():
            with open(id2label_path, "rb") as f:
                self.id2label = pickle.load(f)
        else:
            self.id2label = BILSTM_ID2LABEL
        self.model = BILSTM_Model(len(self.vocab2index), 300, 256, 4)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=torch.device("cpu"))
        )
        self.model.eval()

    def encode_sentence(self, text, vocab2index, N=70):
        tokenized = word_tokenize(text)
        encoded = np.zeros(N, dtype=int)
        enc1 = np.array(
            [vocab2index.get(word, vocab2index["UNK"]) for word in tokenized]
        )
        length = min(N, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded, length

    def predict(self, text: str):
        preprocess_text = preprocess(text)
        x, len_ = self.encode_sentence(preprocess_text, self.vocab2index)
        x, len_ = torch.Tensor(x).long(), torch.Tensor([len_]).long()
        with torch.no_grad():
            logits = self.model(x.unsqueeze(0), len_)[0]
            prob = nn.functional.softmax(logits, dim=-1).tolist()
            pred_class = int(logits.argmax().item())
        return self.id2label[pred_class], prob
