# ◆ Cortex Coder

> An intelligent, self-hosted AI coding assistant powered by a fine-tuned Qwen2.5-Coder model, served via a LangGraph agentic pipeline, FastAPI backend, and a sleek dark-mode web UI.

<div align="center">

[![Live App](https://img.shields.io/badge/Live%20App-Cortex%20Coder-blue?style=for-the-badge)](https://junaidariie.github.io/Cortex-Coder/)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-orange?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/junaid17/Cortex-Coder/tree/main)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-Model-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/junaid17/qwen2.5-0.5b-coding-assistant-gguf)

</div>

---

> **Note on Performance:** Due to the CPU-only inference setup, the **first token may take a few seconds** to appear. After that, generation runs at a comfortable medium pace — not too slow, not too fast. This is expected behavior for a quantized GGUF model running on CPU via `llama-cpp-python`.

---

## Demo

> 🎬 **Video Demo:** *(Coming soon — link will be added here)*

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Fine-Tuning](#fine-tuning)
- [Evaluation Results](#evaluation-results)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [UI Features](#ui-features)
- [Model Details](#model-details)

---

## Overview

Cortex Coder is a full-stack AI coding assistant that:

- Runs a **quantized GGUF model** locally via `llama-cpp-python`
- Uses a **LangGraph multi-node agent** to classify intent, detect language, generate code, and auto-repair bad outputs
- Exposes a **FastAPI streaming backend** with real-time token streaming
- Serves a **pure HTML/CSS/JS frontend** with chat history, code highlighting, copy buttons, and response metadata

---

## Architecture

### Backend API (FastAPI)

```
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend (app.py)                │
│                                                         │
│   GET  /                  →  Health check               │
│   POST /chat/stream       →  StreamingResponse          │
│                               (token-by-token)          │
│   POST /chat/metadata     →  JSON metadata              │
│                               (intent, task, language)  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph Agent (scripts/main.py)          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         ChatLlamaCpp (scripts/load_model.py)            │
│   qwen2.5-0.5b-coding-assistant-q4_k_m.gguf            │
│   n_ctx=4096 | temp=0.7 | max_tokens=1000              │
└─────────────────────────────────────────────────────────┘
```

The frontend (`index.html`) connects to the backend via `fetch` + `ReadableStream` for real-time token streaming. No framework — pure vanilla JS.

---

## Agent Pipeline

The LangGraph graph processes every message through a structured multi-node pipeline. Below is the actual graph visualization:

![LangGraph Agent Pipeline](assets/langgraph_pipeline.png)

### Node Descriptions

| Node | Role |
|---|---|
| `classify_intent` | Keyword-based router: `coding` or `general_chat` |
| `handle_general_chat` | Responds using `GENERAL_CHAT_SYSTEM_PROMPT` |
| `classify_task` | Detects: `code_generation`, `debugging`, or `explanation` |
| `detect_language` | Detects: Python, JavaScript, C++, Java, SQL, or unknown |
| `generate_code` | Calls LLM with task + language context |
| `classify_output` | Checks if output contains valid code markers |
| `repair_code` | Re-prompts LLM to fix malformed output (max 2 retries) |

---

## Project Structure

```
FINAL-CODER/
│
├── app.py                          # FastAPI app — streaming + metadata endpoints
│
├── scripts/
│   ├── main.py                     # LangGraph agent graph definition
│   ├── load_model.py               # Singleton ChatLlamaCpp model loader
│   └── prompts.py                  # System prompts for all agent nodes
│
├── Model/
│   └── qwen2.5-0.5b-coding-assistant-q4_k_m.gguf   # Quantized GGUF model
│
├── Notebook/
│   └── Qwen2_5_small_finetuning.ipynb               # Fine-tuning notebook (Unsloth + LoRA)
│
├── assets/                         # Images for README
├── index.html                      # Frontend — single-file chat UI
├── Dockerfile                      # Container definition (Python 3.10, port 7860)
├── requirements.txt                # Python dependencies
└── myvenv/                         # Local virtual environment
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Runtime | `llama-cpp-python` (CPU inference) |
| LLM Interface | `langchain-community` → `ChatLlamaCpp` |
| Agent Framework | `LangGraph` (stateful multi-node graph) |
| Backend | `FastAPI` + `uvicorn` |
| Streaming | `StreamingResponse` + `ReadableStream` (browser) |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Fine-Tuning | `Unsloth` + `PEFT` + `TRL` (LoRA on Qwen2.5-Coder) |
| Containerization | Docker (Python 3.10 base) |
| Model Hosting | HuggingFace Hub (GGUF format) |

---

## Fine-Tuning

The model was fine-tuned using the notebook at `Notebook/Qwen2_5_small_finetuning.ipynb`.

### Base Model
```
unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit
```

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (`r`) | 32 |
| Alpha (`lora_alpha`) | 64 |
| Dropout | 0.05 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Gradient Checkpointing | `unsloth` (2x memory efficient) |
| Quantization | 4-bit (bnb) |
| Max Sequence Length | 2048 |

### Fine-Tuning Stack

```
torch + transformers + datasets + accelerate
peft + bitsandbytes + trl + unsloth + unsloth_zoo
evaluate + sacrebleu + rouge-score
```

### Training Flow

```
Load base model (4-bit quantized)
        ↓
Apply LoRA adapters (PEFT)
        ↓
Load & split dataset
        ↓
SFTTrainer (TRL) with TrainingArguments
        ↓
Save fine-tuned adapter weights
        ↓
Export to GGUF (Q4_K_M quantization)
        ↓
Upload to HuggingFace Hub
```

---

## Evaluation Results

The fine-tuned model was evaluated against the base model across 5 key metrics. Despite being only a **0.5B parameter model**, the fine-tuning produced significant improvements across all metrics.

### Base vs Fine-tuned Model Performance

![Base vs Fine-tuned Model Performance](assets/eval_comparison.png)

| Metric | Base Model | Fine-tuned Model | Direction |
|---|---|---|---|
| PPL (Perplexity) | 3.58 | 1.98 | ↓ Lower is better |
| BLEU | 13.98 | 24.55 | ↑ Higher is better |
| ROUGE-1 | 0.45 | 0.53 | ↑ Higher is better |
| ROUGE-L | 0.27 | 0.44 | ↑ Higher is better |
| Pass@3 | 14.00 | 56.00 | ↑ Higher is better |

### Percentage Improvement over Base Model

![Percentage Improvement](assets/eval_improvement.png)

| Metric | Improvement |
|---|---|
| PPL | 44.69% reduction |
| BLEU | 75.58% increase |
| ROUGE-1 | 19.33% increase |
| ROUGE-L | 61.55% increase |
| Pass@K | **300.00% increase** |

> The **Pass@3 score jumping from 14 to 56** (a 300% improvement) is the most significant result — meaning the fine-tuned model generates correct, executable code solutions 4x more often than the base model.

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- The GGUF model file placed at `Model/qwen2.5-0.5b-coding-assistant-q4_k_m.gguf`
- Download model from: [junaid17/qwen2.5-0.5b-coding-assistant-gguf](https://huggingface.co/junaid17/qwen2.5-0.5b-coding-assistant-gguf)

### 1. Create virtual environment

```bash
python -m venv myvenv
myvenv\Scripts\activate        # Windows
source myvenv/bin/activate     # Linux/Mac
```

### 2. Install llama-cpp-python (CPU)

```bash
pip install llama-cpp-python==0.3.22 \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```
llama-cpp-python
uvicorn
fastapi
pydantic
langchain
langgraph
langchain-community
python-multipart
```

---

## Running the App

### Start the backend

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

### Open the frontend

Open `index.html` directly in your browser, or visit the live app:

**Live App:** [https://junaidariie.github.io/Cortex-Coder/](https://junaidariie.github.io/Cortex-Coder/)

To run locally, change `API_BASE` in `index.html`:
```javascript
const API_BASE = "http://localhost:7860";
```

---

## Docker Deployment

### Build

```bash
docker build -t cortex-coder .
```

### Run

```bash
docker run -p 7860:7860 cortex-coder
```

### Dockerfile Summary

```dockerfile
FROM python:3.10
WORKDIR /app

# System deps for llama-cpp-python compilation
RUN apt-get install -y build-essential gcc g++ cmake git curl

# Install llama-cpp-python from CPU wheel index
RUN pip install llama-cpp-python==0.3.22 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Install remaining deps
RUN pip install -r requirements.txt

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## API Reference

### `POST /chat/stream`

Streams the assistant response token by token.

**Request**
```json
{
  "message": "Write a Python function to reverse a string",
  "thread_id": "thread_1234567890_abc123"
}
```

**Response**

`text/plain` stream — raw tokens as they are generated.

---

### `POST /chat/metadata`

Returns agent metadata for the last completed turn.

**Request**
```json
{
  "thread_id": "thread_1234567890_abc123"
}
```

**Response**
```json
{
  "intent": "coding",
  "task_type": "code_generation",
  "language": "python",
  "is_code": true,
  "retry_count": 0
}
```

---

### `GET /`

Health check.

```json
{ "message": "Cortex AI API is running" }
```

---

## UI Features

| Feature | Description |
|---|---|
| Real-time streaming | Tokens appear as they are generated with a blinking cursor |
| Code blocks | Syntax-highlighted blocks with language label and Copy button |
| Chat history | Persisted in `localStorage`, survives page refresh |
| Multiple chats | Create, switch, and delete independent chat sessions |
| Response metadata | Shows tokens, speed (t/s), duration, intent, task, language |
| Stop generation | Abort button cancels the stream mid-response |
| Retry | Re-generates the last assistant response |
| Copy / Share | Copy full response or share via Web Share API |
| Like / Dislike | Feedback buttons on each assistant message |
| Suggestion chips | Quick-start prompts on the welcome screen |
| Responsive | Mobile-friendly with collapsible sidebar |
| Dark theme | Full dark UI with `JetBrains Mono` for code |

---

## Model Details

| Property | Value |
|---|---|
| Base Model | Qwen2.5-Coder-0.5B-Instruct |
| Format | GGUF (Q4_K_M quantization) |
| Parameters | ~0.5B |
| Context Length | 4096 tokens |
| Inference | CPU via llama-cpp-python |
| Temperature | 0.7 |
| Max Output Tokens | 1000 |
| HuggingFace Model | [junaid17/qwen2.5-0.5b-coding-assistant-gguf](https://huggingface.co/junaid17/qwen2.5-0.5b-coding-assistant-gguf) |
| HuggingFace Space | [junaid17/Cortex-Coder](https://huggingface.co/spaces/junaid17/Cortex-Coder/tree/main) |

---

## System Prompts

Three specialized prompts drive the agent behavior:

- **`GENERAL_CHAT_SYSTEM_PROMPT`** — Conversational assistant for non-coding queries
- **`CODE_AGENT_SYSTEM_PROMPT`** — Expert coding assistant for generation, debugging, and explanation
- **`REPAIR_SYSTEM_PROMPT`** — Strict repair agent that converts malformed output into valid executable code

---

*Built by Junaid — Cortex Coder v1.0*
