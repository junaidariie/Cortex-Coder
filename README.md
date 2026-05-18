# ◆ Cortex Coder

> An intelligent, self-hosted AI coding assistant powered by a fine-tuned Qwen2.5-Coder model, served via a LangGraph agentic pipeline, FastAPI backend, and a sleek dark-mode web UI.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Fine-Tuning](#fine-tuning)
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

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (index.html)                │
│   ┌──────────┐   ┌──────────────────────────────────┐   │
│   │ Sidebar  │   │         Chat Area                │   │
│   │ - History│   │  - Streaming messages            │   │
│   │ - New    │   │  - Code blocks + copy            │   │
│   │   Chat   │   │  - Metadata bar (tokens, speed)  │   │
│   └──────────┘   └──────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP (fetch + ReadableStream)
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend (app.py)                │
│                                                         │
│   POST /chat/stream   →  StreamingResponse              │
│   POST /chat/metadata →  JSON metadata                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph Agent (scripts/main.py)          │
│                                                         │
│   classify_intent → route                               │
│        ├── general_chat → END                           │
│        └── coding                                       │
│               ├── classify_task                         │
│               ├── detect_language                       │
│               ├── generate_code                         │
│               ├── classify_output                       │
│               │       ├── final → END                   │
│               │       └── repair → classify_output      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         ChatLlamaCpp (scripts/load_model.py)            │
│   qwen2.5-0.5b-coding-assistant-q4_k_m.gguf            │
│   n_ctx=4096 | temp=0.7 | max_tokens=1000              │
└─────────────────────────────────────────────────────────┘
```

---

## Agent Pipeline

The LangGraph graph processes every message through a structured pipeline:

```
                    ┌─────────────────┐
                    │  User Message   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ classify_intent │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────▼────────┐          ┌─────────▼────────┐
     │  general_chat   │          │  classify_task   │
     │  (chat prompt)  │          │  (debug/explain/ │
     └────────┬────────┘          │   code_gen)      │
              │                   └─────────┬────────┘
             END                            │
                                  ┌─────────▼────────┐
                                  │ detect_language  │
                                  │ (py/js/cpp/java/ │
                                  │  sql/unknown)    │
                                  └─────────┬────────┘
                                            │
                                  ┌─────────▼────────┐
                                  │  generate_code   │
                                  │  (LLM call)      │
                                  └─────────┬────────┘
                                            │
                                  ┌─────────▼────────┐
                                  │ classify_output  │
                                  │ (is it code?)    │
                                  └─────────┬────────┘
                                            │
                             ┌──────────────┴──────────────┐
                             │                             │
                    ┌────────▼────────┐          ┌─────────▼────────┐
                    │     final       │          │     repair       │
                    │     END         │          │  (retry ≤ 2x)    │
                    └─────────────────┘          └─────────┬────────┘
                                                           │
                                                  ┌────────▼────────┐
                                                  │ classify_output │
                                                  └─────────────────┘
```

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

## Setup & Installation

### Prerequisites

- Python 3.10+
- The GGUF model file placed at `Model/qwen2.5-0.5b-coding-assistant-q4_k_m.gguf`

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

Open `index.html` directly in your browser, or serve it statically.

The frontend connects to:
```
https://junaid17-cortex-coder.hf.space
```

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
| HuggingFace Repo | `junaid17/qwen2.5-coder-3b-gguf` |

---

## System Prompts

Three specialized prompts drive the agent behavior:

- **`GENERAL_CHAT_SYSTEM_PROMPT`** — Conversational assistant for non-coding queries
- **`CODE_AGENT_SYSTEM_PROMPT`** — Expert coding assistant for generation, debugging, and explanation
- **`REPAIR_SYSTEM_PROMPT`** — Strict repair agent that converts malformed output into valid executable code

---

*Built by Junaid — Cortex Coder v1.0*
