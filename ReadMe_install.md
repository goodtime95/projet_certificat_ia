# Instructions d'installation

## Environment set up

```
bash
python -m venv env
source env/bin/activate
pip install openai
pip install faiss-cpu
pip install requests
pip install pymupdf4llm
pip install -U langgraph langchain-openai langchain-core
pip install langchain-mcp-adapters mcp-server-sqlite
```

## Installation de Ollama sur Onyxia

```
sudo apt-get update
sudo apt-get install zstd
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve & # runs in background
```

## Installation de `tesseract` sur Onyxia (pour le parsing de PDFs)

```
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
```