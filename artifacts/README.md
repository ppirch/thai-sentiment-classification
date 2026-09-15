# Artifacts

Committed:
- `vocab2index.pkl` — BiLSTM vocab (from sentence-classification-train / sentiment-classification-api)
- `bilstm_id2label.pkl`, `bilstm_label2id.pkl` — BiLSTM mapping `{0:neg,1:neu,2:pos,3:q}`
- `stacked_bilstm_model.pth` (40MB, from sentiment-classification-api/weights)

Not committed:
- WangchanBERTa checkpoints — saved by `src/train/*` to `test_trainer/` (gitignored).
