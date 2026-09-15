"""Custom PyTorch training loop for WangchanBERTa.

Migrated from sentence-classification/train_pytorch.py with imports fixed.
Usage: python -m src.train.train_pytorch --data data/train.csv --epochs 5
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from src.config import (
    WANGCHANBERTA_LABEL2ID,
    WANGCHANBERTA_MODEL_NAME,
)
from src.models.wangchanberta import (
    WangchanBERTaClassifier,
    WangchanBERTaDataset,
    load_tokenizer,
)


def train(model, train_data, val_data, tokenizer, learning_rate, epochs):
    train_ds = WangchanBERTaDataset(train_data, tokenizer, WANGCHANBERTA_LABEL2ID)
    val_ds = WangchanBERTaDataset(val_data, tokenizer, WANGCHANBERTA_LABEL2ID)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=24, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch_num in range(epochs):
        model.train()
        total_acc_train = total_loss_train = 0
        for train_input, train_label in tqdm(train_loader):
            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)
            output = model(input_id, mask)
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            total_acc_train += (output.argmax(dim=1) == train_label).sum().item()
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        model.eval()
        total_acc_val = total_loss_val = 0
        with torch.no_grad():
            for val_input, val_label in val_loader:
                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)
                output = model(input_id, mask)
                total_loss_val += criterion(output, val_label.long()).item()
                total_acc_val += (output.argmax(dim=1) == val_label).sum().item()

        print(
            f"Epoch {epoch_num+1} | Train loss {total_loss_train/len(train_data):.3f} "
            f"acc {total_acc_train/len(train_data):.3f} | Val loss {total_loss_val/len(val_data):.3f} "
            f"acc {total_acc_val/len(val_data):.3f}"
        )


def evaluate(model, test_data, tokenizer):
    test_ds = WangchanBERTaDataset(test_data, tokenizer, WANGCHANBERTA_LABEL2ID)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total = 0
    with torch.no_grad():
        for inp, label in loader:
            label = label.to(device)
            mask = inp["attention_mask"].to(device)
            input_id = inp["input_ids"].squeeze(1).to(device)
            total += (model(input_id, mask).argmax(dim=1) == label).sum().item()
    print(f"Test accuracy: {total/len(test_data):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()

    tokenizer = load_tokenizer(WANGCHANBERTA_MODEL_NAME)
    df = pd.read_csv(args.data)
    np.random.seed(112)
    df_train, df_val, df_test = np.split(
        df.sample(frac=1, random_state=42),
        [int(0.8 * len(df)), int(0.9 * len(df))],
    )
    model = WangchanBERTaClassifier()
    train(model, df_train, df_val, tokenizer, args.lr, args.epochs)
    evaluate(model, df_test, tokenizer)
