# thai-sentiment-classification

Thai Wisesight sentiment classification (`pos` / `neu` / `neg` / `q`).

Consolidated from 3 repos (clean migrate, histories not preserved):
- `sentence-classification` — WangchanBERTa training (PyTorch + HF Trainer), `get_wisesight_sentiment.py`, `data/train.csv`
- `sentiment-classification-api` — FastAPI + BiLSTM serving (`model.py`, `main.py`, Dockerfile, `weights/stacked_bilstm_model.pth`)
- `sentence-classification-train` — BiLSTM/HF training notebooks, raw `data/*.txt`, `dumps/*.pkl`

## Layout
```
src/
  config.py          # label maps (note: BiLSTM vs BERT ids differ!), model names
  data.py            # Wisesight download -> CSV
  preprocess.py      # Thai cleanup for BiLSTM
  models/
    bilstm.py        # BILSTM_Model + BiLSTMSentimentModel (was MySentimentModel)
    wangchanberta.py # WangchanBERTaClassifier + Dataset
  train/
    train_pytorch.py # custom loop (was train_pytorch.py)
    train_hf.py      # Trainer API (was train_hugging_face.py)
  api/
    main.py          # FastAPI /predict (was sentiment-classification-api/main.py)
scripts/download_data.py
notebooks/           # 4 archived notebooks, as-is
artifacts/           # vocab + label maps committed; *.pth gitignored (see artifacts/README.md)
data/                # gitignored; regenerate via script
Dockerfile, docker-compose.yml, requirements.txt
```

## Quickstart
```bash
pip install -r requirements.txt
python scripts/download_data.py --out-dir data

# train BERT (custom loop)
python -m src.train.train_pytorch --data data/train.csv --epochs 5
# or HF Trainer
python -m src.train.train_hf --data data/train.csv

# serve BiLSTM (needs artifacts/stacked_bilstm_model.pth restored)
uvicorn src.api.main:app --host 0.0.0.0 --port 8080
# docker
docker compose up --build
```

## Notes / breaking changes
- Label ids differ historically: BiLSTM `{neg:0, neu:1, pos:2, q:3}`, WangchanBERTa `{pos:0, neu:1, neg:2, q:3}`. Kept as-is in `src/config.py` — unify in future.
- `weights/stacked_bilstm_model.pth` (40MB) and `data/*.csv` are NOT committed; restore from backup/Release.
- Notebooks in `notebooks/` are archived, not refactored.
