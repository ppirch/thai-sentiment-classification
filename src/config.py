"""Shared config. Note: BiLSTM and WangchanBERTa use different label ids historically."""

WANGCHANBERTA_MODEL_NAME = "airesearch/wangchanBERTa-base-att-spm-uncased"

# sentence-classification (WangchanBERTa, 2023)
WANGCHANBERTA_LABEL2ID = {"pos": 0, "neu": 1, "neg": 2, "q": 3}
WANGCHANBERTA_ID2LABEL = {v: k for k, v in WANGCHANBERTA_LABEL2ID.items()}

# sentiment-classification-api / sentence-classification-train (BiLSTM, 2021)
BILSTM_LABEL2ID = {"neg": 0, "neu": 1, "pos": 2, "q": 3}
BILSTM_ID2LABEL = {v: k for k, v in BILSTM_LABEL2ID.items()}

MAX_LEN_BERT = 416
MAX_LEN_BILSTM = 70
