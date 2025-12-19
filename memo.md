# LayoutLLM-T2I 実行メモ

## 環境
- サーバー:
- 作業dir:
- GPU: (nvidia-smi貼る)
- conda env: layoutllm_t2i
- Python:
- torch/cuda:

## 取得物
- policy ckpt: checkpoints/policy/...
- diffusion ckpt: checkpoints/diffusion/...
- candidates: data/candidates/...

## 実行コマンド
```bash
export OPENAI_API_KEY=...
python txt2img.py ...
```

## 結果
- 出力先:
- 生成画像:
- ログ:

## 詰まり・対応
- 症状:
- エラー:
- 対処:
- 次やる:

## 次の一歩（最短で「動作確認」する順番）
1) `nvidia-smi`  
2) `conda env` 作成 & `pip install -r requirements.txt`  
3) `torch.cuda.is_available()`  
4) `ckpt / candidate` を所定位置に置く  
5) `OPENAI_API_KEY` export  
6) `txt2img.py` を短い prompt で1回回す  
7) `generation_samples/` と `logs/` を確認