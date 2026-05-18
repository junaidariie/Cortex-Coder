# model_loader.py
from langchain_community.chat_models import ChatLlamaCpp
import os

_llm_instance = None

def get_model(callbacks=None):
    global _llm_instance
    if _llm_instance is None:
        model_path = os.path.join(os.path.dirname(__file__), os.pardir, "Model", "qwen2.5-0.5b-coding-assistant-q4_k_m.gguf")
        model_path = os.path.abspath(model_path)
        print(f"Loading ChatLlamaCpp model from local path: {model_path}")
        _llm_instance = ChatLlamaCpp(
            model_path=model_path,
            temperature=0.7,
            max_tokens=1000,
            n_ctx=4096,
            n_batch=512,
            n_threads=8,
            n_gpu_layers=0,
            verbose=False,
        )
        print("Model loaded successfully!")
    return _llm_instance
