"""HuggingFace Trainer path. Migrated from sentence-classification/train_hugging_face.py."""
import numpy as np
import pandas as pd
from datasets import Dataset
from torch.optim import AdamW, lr_scheduler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import evaluate

from src.config import WANGCHANBERTA_LABEL2ID, WANGCHANBERTA_MODEL_NAME

metric = evaluate.load("accuracy")
MODEL_NAME = WANGCHANBERTA_MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_dataset(data):
    return tokenizer(data["text"], max_length=416, truncation=True, padding="max_length")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return metric.compute(predictions=np.argmax(logits, axis=-1), references=labels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--output", default="test_trainer")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["label"] = df["label"].map(WANGCHANBERTA_LABEL2ID)

    np.random.seed(112)
    df_train, df_val, df_test = np.split(
        df.sample(frac=1, random_state=42),
        [int(0.8 * len(df)), int(0.9 * len(df))],
    )

    def make_ds(d):
        ds = Dataset.from_pandas(d).map(tokenize_dataset, batched=True)
        cols = [c for c in ["text", "__index_level_0__"] if c in ds.column_names]
        if cols:
            ds = ds.remove_columns(cols)
        return ds.rename_column("label", "labels")

    train_ds, eval_ds = make_ds(df_train), make_ds(df_val)
    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output,
            evaluation_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
    )
    trainer.train()
