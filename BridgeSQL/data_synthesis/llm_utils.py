"""
LLM and Embedding API configuration.

All models are called via the OpenAI-compatible API. Configure your model
endpoints and API keys in MODEL_CONFIG below before running the pipeline.
"""

from openai import OpenAI, RateLimitError, APITimeoutError

# ============================================================================
# USER CONFIGURATION — fill in your endpoints and API keys here
# ============================================================================

MODEL_CONFIG = {
    # Embedding model (used in filter_databases.py)
    "text-embedding-v4": {
        "base_url": "https://your-embedding-endpoint/v1",
        "api_key": "your-api-key",
    },
    # Chat models (used in database_population.py and peer_review.py)
    "qwen3-max": {
        "base_url": "https://your-chat-endpoint/v1",
        "api_key": "your-api-key",
    },
    "kimi-k2.5": {
        "base_url": "https://your-chat-endpoint/v1",
        "api_key": "your-api-key",
    },
}

# Embedding model name (must be a key in MODEL_CONFIG)
EMBEDDING_MODEL = "text-embedding-v4"

# Chat models used for data synthesis (randomly sampled per request)
CHAT_MODELS = ["qwen3-max", "kimi-k2.5"]

# ============================================================================


class LLMClient:
    """Unified LLM client driven by MODEL_CONFIG."""

    def __init__(self):
        self._clients = {}

    @property
    def chat_models(self):
        return list(CHAT_MODELS)

    @property
    def embedding_model(self):
        return EMBEDDING_MODEL

    def _get_client(self, model: str) -> OpenAI:
        if model not in MODEL_CONFIG:
            raise ValueError(
                f"Model '{model}' not in MODEL_CONFIG. "
                f"Available: {list(MODEL_CONFIG.keys())}"
            )
        if model not in self._clients:
            cfg = MODEL_CONFIG[model]
            self._clients[model] = OpenAI(
                api_key=cfg["api_key"],
                base_url=cfg["base_url"],
                timeout=cfg.get("timeout", 30),
            )
        return self._clients[model]

    def call(self, model: str, prompt: str, temperature: float = 0.7) -> str | None:
        """Call a chat model and return the response text, or None on failure."""
        client = self._get_client(model)
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except RateLimitError:
            print(f"[WARN] Rate limit: {model}")
            return None
        except APITimeoutError:
            print(f"[WARN] Timeout: {model}")
            return None
        except Exception as e:
            print(f"[ERROR] {model}: {e}")
            return None

    def embed(self, text: str) -> list[float] | None:
        """Compute a text embedding vector using the configured embedding model."""
        model = EMBEDDING_MODEL
        client = self._get_client(model)
        try:
            resp = client.embeddings.create(model=model, input=text)
            return resp.data[0].embedding
        except RateLimitError:
            print(f"[WARN] Rate limit: {model}")
            return None
        except APITimeoutError:
            print(f"[WARN] Timeout: {model}")
            return None
        except Exception as e:
            print(f"[ERROR] embed: {e}")
            return None
