# general-purpose RAG

> ローカル完結型の汎用 RAG（検索拡張生成）システム

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit)

---

## 概要
PDF・TXT ファイルをアップロードするだけで、自然言語でドキュメントへ問い合わせができるシステムです。  
外部 API を一切使わず、**すべての処理をローカル環境で完結**させています。

---

## システム構成

### ドキュメント登録フロー（ベクトルDB 構築）
![](./docs/img/save_DB.png)

### クエリ投入フロー（回答生成）
![](./docs/img/answer_query.png)

---

## 主な機能

- **多言語対応** — 日本語・英語・中国語に対応した埋め込みモデル (`intfloat/multilingual-e5`)
- **差分更新** — MD5 ハッシュによる重複検出でファイルの再インデックスを防止
- **ストリーミング応答** — トークン単位のリアルタイム出力
- **ソース帰属** — 回答に参照元ファイルを表示
- **ローカル完結** — 外部 API 不要、インターネット接続なしで動作
- **柔軟なベクトルDB** — ChromaDB / FAISS を差し替え可能（抽象基底クラスで設計）

---

## 技術スタック

| カテゴリ | ライブラリ | 用途 |
|----------|-----------|------|
| バックエンド | FastAPI + Uvicorn | REST API サーバー |
| RAG パイプライン | LangChain | ドキュメント処理・LLM 連携 |
| 埋め込み | sentence-transformers | テキストのベクトル化 |
| ベクトルDB | ChromaDB / FAISS | 類似度検索 |
| LLM | Ollama | ローカル LLM 推論 |
| フロントエンド | Streamlit | Web UI |
| ドキュメント処理 | PyPDF | PDF テキスト抽出 |
| テスト | pytest / RAGAS | ユニット・統合・評価テスト |

---

## 必要条件

**Docker を使う場合（推奨）**
- Docker Desktop 24.0+
- Docker Compose v2

**スタンドアロンで動かす場合**
- Python 3.10+
- [Ollama](https://ollama.com/) のインストールと起動

---

## セットアップ

### Docker で起動（推奨）

```bash
# リポジトリをクローン
git clone <REPO_URL>
cd general-purpose_RAG

# コンテナのビルドと起動
docker compose up --build -d

# LLM モデルをダウンロード（初回のみ）
docker exec -it general-purpose_rag-ollama-1 ollama pull qwen2.5:3b

# API の起動確認
curl http://localhost:8000/health
```

ブラウザで http://localhost:8501 を開くと UI が表示されます。

### スタンドアロンで起動

```bash
# 依存パッケージをインストール
pip install -r requirements.txt

# ドキュメントからベクトルDBを構築
python dataset_maker.py --source local --path ./docs_storage

# CLI でクエリを実行
python query_answer.py --model qwen2.5:3b --k 5
```

---

## 使い方

1. **ドキュメントをアップロード** — UI の「ファイルアップロード」から PDF または TXT ファイルを選択
2. **質問を入力** — テキストボックスに自然言語で質問を入力して送信
3. **回答を確認** — LLM の回答と、参照元ファイル名が表示されます

API を直接使う場合:

```bash
# クエリ（同期）
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "このドキュメントの要点は？", "k": 5}'

# ファイル一覧
curl http://localhost:8000/documents
```

---

## 設定

環境変数で動作をカスタマイズできます。

| 変数名 | デフォルト値 | 説明 |
|--------|------------|------|
| `LLM_MODEL` | `qwen2.5:1.5b` | 使用する Ollama モデル |
| `EMBED_MODEL` | `intfloat/multilingual-e5-small` | 埋め込みモデル |
| `CHROMA_DIR` | `./chroma_db` | ChromaDB の保存先 |
| `DOCS_DIR` | `./docs_storage` | ドキュメントの保存先 |
| `API_URL` | `http://localhost:8000` | フロントエンドが参照する API URL |

---

## ディレクトリ構成

```
general-purpose_RAG/
├── src/                  # RAG コアパイプライン
│   ├── api.py            # FastAPI エンドポイント
│   ├── dataload.py       # ドキュメント読み込み
│   ├── make_chunk.py     # テキスト分割
│   ├── embedding.py      # ベクトル埋め込み
│   ├── vector_store.py   # ChromaDB / FAISS
│   ├── retriever.py      # 検索 + LLM 連携
│   └── llm.py            # Ollama ラッパー
├── ui/                   # Streamlit フロントエンド
│   └── app.py
├── tests/                # テストスイート
│   ├── unit/
│   ├── integration/
│   └── evaluation/
├── docs/                 # 設計ドキュメント
├── docs_storage/         # アップロードされたドキュメント
├── chroma_db/            # ベクトルDB（永続化）
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## テスト

```bash
# ユニットテスト
pytest tests/unit/

# 統合テスト
pytest tests/integration/

# RAG 評価
pip install -r requirements-eval.txt
pytest tests/evaluation/
```
