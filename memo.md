# LayoutLLM-T2I 実行メモ

## 処理概要
```
- テキスト
  ↓
- LLM（GPT系）
  ↓
- Policy Network  ←★ ここが policy_weights.pt
  - 「再学習してね」ではなく「この重みを使って実験を再現してね」タイプの研究コード
  ↓
- レイアウト（候補選択・配置）
  ↓
- Diffusion（GLIGEN系）
  ↓
- 画像生成
```

```
pwd
/home/takehara/Projects/LayoutLLM-T2I
```

### GPU が見えているか確認
```
takehara@yoshitakalab-Super-Server:~/Projects/LayoutLLM-T2I$ nvidia-smi
Fri Dec 19 14:21:36 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.142                Driver Version: 550.142        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        On  |   00000000:17:00.0 Off |                  N/A |
| 30%   26C    P8              9W /  350W |     261MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        On  |   00000000:65:00.0 Off |                  N/A |
| 30%   28C    P8              7W /  350W |       2MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      1110      G   /usr/lib/xorg/Xorg                             39MiB |
|    0   N/A  N/A      1197      G   /usr/bin/gnome-shell                           74MiB |
|    0   N/A  N/A      2090      G   /usr/lib/xorg/Xorg                             18MiB |
|    0   N/A  N/A      2221      G   /usr/bin/gnome-shell                          105MiB |
+-----------------------------------------------------------------------------------------+
takehara@yoshitakalab-Super-Server:~/Projects/LayoutLLM-T2I$ 
```

### Python / conda の状態確認
```
takehara@yoshitakalab-Super-Server:~/Projects/LayoutLLM-T2I$ which python
/usr/bin/python
takehara@yoshitakalab-Super-Server:~/Projects/LayoutLLM-T2I$ python --version
Python 3.9.9
takehara@yoshitakalab-Super-Server:~/Projects/LayoutLLM-T2I$ conda --version
bash: conda: command not found
takehara@yoshitakalab-Super-Server:~/Projects/LayoutLLM-T2I$ 
```

conda が入っていない（conda: command not found）なので、 README の “conda create …” ルートではなく、まずは venv（Python標準）で環境を作る。

### sudo
できない

### 環境構築

#### 0. 前提条件・制約の整理（重要）
- 研究室GPUサーバー
  - sudo 不可
  - ユーザーホーム配下は自由に操作可
  - GPU: RTX 3090
  - NVIDIA Driver: 550.x（CUDA 12.4 対応だが、PyTorch側でCUDA同梱を使用）

- このため
  - system Python / apt は使わない
  - micromamba + pip 併用で環境構築

#### 1. micromamba の導入（conda 代替）
- 目的
  - sudoなしで Python / C拡張 / spacy などを安定して入れるため

- 実行コマンド
```
mkdir -p ~/bin
curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest -o /tmp/micromamba.tar.bz2
tar -xjf /tmp/micromamba.tar.bz2 -C /tmp
cp /tmp/bin/micromamba ~/bin/micromamba
chmod +x ~/bin/micromamba
```

```
# shell hook
eval "$(~/bin/micromamba shell hook -s bash)"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
```

#### 2. Python 仮想環境作成
- 理由
  - LayoutLLM-T2I は Python 3.8 前提
  - spacy / torch 互換のため

- 実行コマンド
```
micromamba create -y -n layoutllm_t2i python=3.8 pip
micromamba activate layoutllm_t2i
```

- 確認
```
python --version   # Python 3.8.x
```


#### 3. PyTorch（GPU対応）のインストール
- 理由
  - conda-forge の torch は glibc 依存で失敗
  - **公式 PyTorch wheel（CUDA 11.3 同梱）**を使用

- 実行コマンド
```
pip install \
  torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

- 動作確認
```
python - << 'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```


#### 4. HuggingFace / Diffusers 周り（バージョン固定）
- 理由
  - LayoutLLM-T2I は 古めの diffusers API を使用
  - huggingface_hub の関数削除に注意

- 採用バージョン
| package         | version |
| --------------- | ------- |
| transformers    | 4.30.2  |
| diffusers       | 0.24.0  |
| huggingface_hub | 0.20.3  |
| accelerate      | 0.24.1  |

- 実行コマンド
```
pip install \
  huggingface_hub==0.20.3 \
  transformers==4.30.2 \
  diffusers==0.24.0 \
  accelerate==0.24.1
```


#### 5. pip バージョン固定（重要）
- 問題
  - pip 25.x は pytorch-lightning 1.x を拒


- 対処
```
pip install pip==24.0
pip --version
```


#### 6. pytorch-lightning & torchmetrics
- 理由
  - PolicyNetwork の aesthetic model が Lightning 依存

- 互換バージョン
| package           | version |
| ----------------- | ------- |
| pytorch-lightning | 1.7.7   |
| torchmetrics      | 0.9.3   |


- 実行コマンド
```
pip install pytorch-lightning==1.7.7
pip install torchmetrics==0.9.3
```


#### 7. NLP / Scene Graph 関連
- sng_parser
  - PyPI に無いため GitHub から直接

- 実行コマンド
```
pip install git+https://github.com/vacancy/SceneGraphParser.git
```

- spacy
  - pip build は gcc が無く失敗
  - micromamba 経由でインストール済み（内部依存解決済）


#### 8. CLIP（OpenAI公式）

- 実行コマンド
```
pip install git+https://github.com/openai/CLIP.git
```


#### 9. その他の必須ライブラリ（コード実行時に判明）

- 実行コマンド
```
pip install \
  omegaconf==2.3.0 \
  pandas==1.5.3 \
  backoff \
  tensorboard==2.11.2
```

#### 10. 

- 実行コマンド
```
mkdir -p \
  checkpoints/policy \
  checkpoints/diffusion \
  data/candidates \
  generation_samples \
  logs
```

#### 11. 最終確認（重要）

- 実行コマンド
```
python txt2img.py --help
```
- 正常に help が表示 → 環境構築フェーズ完了


#### まとめ

- sudo 不可 → micromamba 採用
- Python 3.8 固定
- torch は pip + cu113 wheel
- pip は 24.0 固定
- lightning 1.7.7 + torchmetrics 0.9.3
- diffusers / HF は 中間世代で固定
- sng_parser / CLIP は GitHub 直install
- txt2img.py --help が通ればOK


### policy_weights.ptの準備
できた

### 事前学習済み重み（Diffusion）を入手して置く
- READMEでは diffusion 重みの入手先が2つがあるとのこと。
  - Baidu（使えるならそれでもOK）
  - HuggingFace（おすすめ）

#### HuggingFace から落とす（サーバーで実行）　<- おすすめからいく
```
cd ~/Projects/LayoutLLM-T2I
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="leigangqu/LayoutLLM-T2I",
    repo_type="model",
    local_dir="checkpoints/diffusion",
    local_dir_use_symlinks=False,
)
print("डाउनロード先:", path)
PY
```

### ファイル準備（Policy / Diffusion / Candidate）

#### Policy weights（手元PCでDL → scpでサーバーへ）

- 公式リンク（Google Drive）
  - policy weights（著者配布）
  - https://drive.google.com/file/d/1t7M-uqgB5GMATJGEe2sM7oZX_ex4EysE/view?usp=sharing

- GPUサーバーに配置
```
cd ~/Projects/LayoutLLM-T2I
ls -lh checkpoints/policy
# policy_weights.pt が 386K 程度で存在
```

#### Diffusion weights（HuggingFaceからサーバーで直接DL → chunk結合）

- HFからダウンロード
```
cd ~/Projects/LayoutLLM-T2I
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="leigangqu/LayoutLLM-T2I",
    repo_type="model",
    local_dir="checkpoints/diffusion",
    local_dir_use_symlinks=False,
)
print("downloaded to:", path)
PY
```

- ダウンロード結果（chunk）
```
ls -lh checkpoints/diffusion | head -n 50
# checkpoints_00335001_model_chunk0.pth (2.5G)
# checkpoints_00335001_model_chunk1.pth (2.3G)
# checkpoints_00335001_others.pth       (2.3G)
# combine.py
```

- chunk結合（combine.py）
```
cd ~/Projects/LayoutLLM-T2I/checkpoints/diffusion
python combine.py
ls -lh
# checkpoints_00335001.pth (7.0G) が生成されていれば成功
```

#### Candidate example file（手元PCでDL → scp）
- ローカルPCで：
```
scp ~/Downloads/train2014_candidate_32.json \
  takehara@yoshitakalab-Super-Server:/home/takehara/Projects/LayoutLLM-T2I/data/candidates/
```

- サーバーで確認：
```
cd ~/Projects/LayoutLLM-T2I
ls -lh data/candidates
# train2014_candidate_32.json があること
```

#### ここまでの最終チェック（ファイル準備完了チェック）
```
cd ~/Projects/LayoutLLM-T2I
ls -lh checkpoints/policy
ls -lh checkpoints/diffusion | head
ls -lh data/candidates
```

期待する状態：
```
checkpoints/policy/policy_weights.pt
checkpoints/diffusion/checkpoints_00335001.pth
data/candidates/train2014_candidate_32.json
```

#### OPENAI_API_KEY の設定
- 環境変数に設定
```
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

- 設定確認
```python - << 'PY'
import os
print("OPENAI_API_KEY set:", os.environ.get("OPENAI_API_KEY") is not None)
PY
```

- txt2img 実行（最小構成・1枚生成）
```
python txt2img.py \
  --gpu 0 \
  --folder generation_samples \
  --prompt "a large clock tower next to a small white church." \
  --policy_ckpt_path checkpoints/policy/policy_weights.pt \
  --diff_ckpt_path checkpoints/diffusion/checkpoints_00335001.pth \
  --cand_path data/candidates/train2014_candidate_32.json \
  --num_per_prompt 1 \
  2>&1 | tee logs/txt2img_run_$(date +%Y%m%d_%H%M%S).log
```

- 各引数の意味（簡易メモ）
  - --gpu 0: 使用する GPU ID
  - --folder generation_samples: 出力先ディレクトリ
  - --prompt: 生成したいテキストプロンプト
  - --policy_ckpt_path: Policy Network の重み
  - --diff_ckpt_path: GLIGEN ベース diffusion モデル（結合済）
  - --cand_path: COCO2014 由来の candidate layout 例
  - --num_per_prompt 1: 1プロンプトにつき1枚（OOM回避）

- 出力確認


### プロンプト
```bash
python txt2img.py \
  --gpu 0 \
  --folder "generation_samples/${RUN_TAG}" \
  --prompt "A man holding a cup while sitting on a chair next to a table with a cat under it" \
  --cand_path data/candidates/train2014_candidate_32.json \
  --policy_ckpt_path checkpoints/policy/policy_weights.pt \
  --diff_ckpt_path checkpoints/diffusion/checkpoints_00335001.pth \
  --config_train_path configs/args_infer.json \
  --num_per_prompt 4 \
  2>&1 | tee "logs/txt2img_${RUN_TAG}.log"
```


### args_infer.json

- 初期値
```json
{
  "exp": "layoutt2i",
  "shot_number": 2,
  "seed": 53,
  "train_json_path": "",
  "feature_path": "",
  "img_dir": "",
  "sampled_data_dir": "",
  "train_number": 64,
  "cand_number": 32,
  "num_workers": 0,
  "engine": "gpt-3.5-turbo",
  "temperature": 0.0,
  "max_tokens": 512,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "gpu": "0",
  "model_config": "openai/clip-vit-large-patch14",
  "lr": 0.0005,
  "epochs": 10,
  "embedding_size": 128,
  "batch_size": 1,
  "policy_temperature": 1.0,
  "ckpt_root": "./checkpoints",
  "aesthetic_ckpt": ""
}
```

- 修正
```json
{
  "exp": "layoutt2i",
  "shot_number": 2,
  "seed": 53,
  "train_json_path": "",
  "feature_path": "",
  "img_dir": "",
  "sampled_data_dir": "",
  "train_number": 64,
  "cand_number": 32,
  "num_workers": 0,
  "engine": "gpt-5.2",
  "temperature": 0.0,
  "max_tokens": 512,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "gpu": "0",
  "model_config": "openai/clip-vit-large-patch14",
  "lr": 0.0005,
  "epochs": 10,
  "embedding_size": 128,
  "batch_size": 1,
  "policy_temperature": 1.0,
  "ckpt_root": "./checkpoints",
  "aesthetic_ckpt": ""
}
```


## 疑問・質問
- `policy_weights.pt`とは
  - 「作者が学習して保存した PolicyNetwork の重み」とのことだが、よくわかっていない。
  - どこで使う？
    - txt2img.pyの中で、いかが呼ばれる。作者が学習済み重みを前提に設計したコード(実験を再現するための前提条件) 
    ```
    PolicyNetwork.load_from_checkpoint(policy_ckpt_path)
    ```
    - 重みなしでは実行不可
    - 自動ダウンロードなし
    - 再学習コードは事実上使えない

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