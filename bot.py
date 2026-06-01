import os
import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, render_template_string, request, send_from_directory
from flask_cors import CORS

APP_NAME = os.getenv("APP_NAME", "Groq AI Backend")
PORT = int(os.getenv("PORT", "10000"))

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()  # optional / reserved for later
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "").strip()

# Comma-separated API keys and models.
# Example:
# GROQ_API_KEYS=key1,key2,key3
# GROQ_MODELS=llama-3.1-8b-instant,gemma2-9b-it,llama-3.3-70b-versatile
GROQ_API_KEYS_RAW = os.getenv("GROQ_API_KEYS", "").strip()
GROQ_MODELS_RAW = os.getenv("GROQ_MODELS", "llama-3.1-8b-instant").strip()

# Optional extra prompt control
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Reply in the same language as the user.",
)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.1-8b-instant").strip()
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "12"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))

DATA_DIR = Path(os.getenv("DATA_DIR", "."))
MODELS_FILE = DATA_DIR / "models.json"
CHAT_FILE = DATA_DIR / "chat_state.json"

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

app = Flask(__name__)
CORS(app)

# -----------------------
# Persistence
# -----------------------
def _safe_read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _safe_write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_models():
    data = _safe_read_json(MODELS_FILE, None)
    if data is not None:
        return data

    api_keys = [k.strip() for k in GROQ_API_KEYS_RAW.split(",") if k.strip()]
    models = [m.strip() for m in GROQ_MODELS_RAW.split(",") if m.strip()]
    items = []
    if api_keys and models:
        # pair first key to all models unless more keys exist
        for i, model in enumerate(models):
            api_key = api_keys[min(i, len(api_keys) - 1)]
            items.append({
                "id": str(uuid.uuid4())[:8],
                "name": model,
                "model": model,
                "api_key": api_key,
                "enabled": True,
                "created_at": int(time.time()),
            })
    else:
        items.append({
            "id": str(uuid.uuid4())[:8],
            "name": DEFAULT_MODEL,
            "model": DEFAULT_MODEL,
            "api_key": "",
            "enabled": True,
            "created_at": int(time.time()),
        })

    _safe_write_json(MODELS_FILE, items)
    return items

def save_models(models):
    _safe_write_json(MODELS_FILE, models)

def load_chat_state():
    return _safe_read_json(CHAT_FILE, {})

def save_chat_state(state):
    _safe_write_json(CHAT_FILE, state)

MODELS = load_models()
CHAT_STATE = load_chat_state()

# -----------------------
# Helpers
# -----------------------
def get_base_url():
    # Works behind Render proxy too
    forwarded_proto = request.headers.get("X-Forwarded-Proto", request.scheme)
    forwarded_host = request.headers.get("X-Forwarded-Host", request.host)
    return f"{forwarded_proto}://{forwarded_host}"

def require_admin():
    if not ADMIN_API_KEY:
        return True
    provided = (
        request.headers.get("X-Admin-Key", "")
        or request.args.get("admin_key", "")
        or (request.json.get("admin_key", "") if request.is_json and isinstance(request.json, dict) else "")
    )
    return provided == ADMIN_API_KEY

def normalize_messages(message: str, history: Optional[List[Dict]] = None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for item in history[-MAX_HISTORY:]:
            role = item.get("role", "user")
            content = str(item.get("content", ""))
            if role in ("user", "assistant", "system") and content.strip():
                msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": message})
    return msgs

def groq_chat(api_key: str, model: str, messages: List[Dict], temperature=0.7, max_tokens=1200) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code >= 400:
        try:
            data = resp.json()
            err = data.get("error", {}).get("message") or data.get("message") or resp.text
        except Exception:
            err = resp.text
        raise RuntimeError(f"HTTP {resp.status_code}: {err}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def is_rate_limit_error(err: Exception) -> bool:
    text = str(err).lower()
    return any(k in text for k in ["429", "rate limit", "too many requests", "quota", "limit"])

def pick_candidates(preferred_model: Optional[str] = None):
    enabled = [m for m in MODELS if m.get("enabled", True)]
    if not enabled:
        return []
    if preferred_model:
        preferred = [m for m in enabled if m.get("model") == preferred_model or m.get("id") == preferred_model]
        others = [m for m in enabled if m not in preferred]
        return preferred + others
    # try default first
    default_first = [m for m in enabled if m.get("model") == DEFAULT_MODEL]
    others = [m for m in enabled if m not in default_first]
    return default_first + others

def touch_chat(chat_id: str, role: str, content: str):
    CHAT_STATE.setdefault(chat_id, [])
    CHAT_STATE[chat_id].append({"role": role, "content": content, "ts": int(time.time())})
    CHAT_STATE[chat_id] = CHAT_STATE[chat_id][-MAX_HISTORY * 2 :]
    save_chat_state(CHAT_STATE)

# -----------------------
# Pages
# -----------------------
@app.get("/")
def home():
    html = Path("index.html")
    if html.exists():
        return send_from_directory(".", "index.html")
    return jsonify({
        "ok": True,
        "name": APP_NAME,
        "message": "index.html not found"
    })

@app.get("/api")
def api_root():
    return jsonify({
        "ok": True,
        "name": APP_NAME,
        "base_url": get_base_url(),
        "default_model": DEFAULT_MODEL,
        "models_count": len([m for m in MODELS if m.get("enabled", True)]),
        "endpoints": {
            "health": "/api/health",
            "info": "/api/info",
            "chat": "/api/chat",
            "models_list": "/api/admin/models",
            "models_add": "/api/admin/models",
            "models_delete": "/api/admin/models/<id>",
            "test": "/api/test",
        }
    })

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "status": "running", "time": int(time.time())})

@app.get("/api/info")
def info():
    return jsonify({
        "ok": True,
        "name": APP_NAME,
        "base_url": get_base_url(),
        "default_model": DEFAULT_MODEL,
        "available_models": [
            {k: v for k, v in m.items() if k != "api_key"}
            for m in MODELS
            if m.get("enabled", True)
        ],
        "chat_cache_keys": list(CHAT_STATE.keys())[:20],
    })

# -----------------------
# Chat endpoints
# -----------------------
@app.post("/api/chat")
def chat():
    if not request.is_json:
        return jsonify({"ok": False, "error": "JSON body required"}), 400

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"ok": False, "error": "message is required"}), 400

    chat_id = str(data.get("chat_id") or data.get("session_id") or "default")
    preferred_model = data.get("model")
    temperature = float(data.get("temperature", 0.7))
    max_tokens = int(data.get("max_tokens", 1200))

    history = CHAT_STATE.get(chat_id, [])
    messages = normalize_messages(message, history=history)

    candidates = pick_candidates(preferred_model=preferred_model)
    if not candidates:
        return jsonify({"ok": False, "error": "No enabled models available"}), 503

    errors = []
    for candidate in candidates:
        api_key = (candidate.get("api_key") or "").strip()
        model = candidate.get("model")
        if not api_key:
            errors.append(f"{candidate.get('name')} missing api_key")
            continue
        try:
            answer = groq_chat(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            touch_chat(chat_id, "user", message)
            touch_chat(chat_id, "assistant", answer)

            return jsonify({
                "ok": True,
                "chat_id": chat_id,
                "model_used": model,
                "provider_name": candidate.get("name"),
                "answer": answer,
                "base_url": get_base_url(),
            })
        except Exception as e:
            errors.append(f"{model}: {e}")
            if not is_rate_limit_error(e):
                # try next provider even on non-rate-limit issues to support failover
                continue

    return jsonify({
        "ok": False,
        "error": "All models failed",
        "details": errors,
    }), 429 if any("rate" in e.lower() or "429" in e.lower() for e in errors) else 502

@app.post("/api/test")
def test_chat():
    # Same as chat, kept for easier frontend testing
    return chat()

# -----------------------
# Admin model management
# -----------------------
@app.get("/api/admin/models")
def admin_list_models():
    if not require_admin():
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    return jsonify({
        "ok": True,
        "models": [
            {k: v for k, v in m.items() if k != "api_key"}
            for m in MODELS
        ]
    })

@app.post("/api/admin/models")
def admin_add_model():
    if not require_admin():
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    if not request.is_json:
        return jsonify({"ok": False, "error": "JSON body required"}), 400

    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    model = str(data.get("model", "")).strip()
    api_key = str(data.get("api_key", "")).strip()
    enabled = bool(data.get("enabled", True))

    if not name or not model or not api_key:
        return jsonify({"ok": False, "error": "name, model, api_key are required"}), 400

    item = {
        "id": str(uuid.uuid4())[:8],
        "name": name,
        "model": model,
        "api_key": api_key,
        "enabled": enabled,
        "created_at": int(time.time()),
    }
    MODELS.append(item)
    save_models(MODELS)

    return jsonify({
        "ok": True,
        "message": "Model added",
        "model": {k: v for k, v in item.items() if k != "api_key"}
    })

@app.delete("/api/admin/models/<model_id>")
def admin_delete_model(model_id: str):
    if not require_admin():
        return jsonify({"ok": False, "error": "Unauthorized"}), 401

    global MODELS
    before = len(MODELS)
    MODELS = [m for m in MODELS if m.get("id") != model_id]
    if len(MODELS) == before:
        return jsonify({"ok": False, "error": "Model not found"}), 404
    save_models(MODELS)
    return jsonify({"ok": True, "message": "Model deleted", "id": model_id})

@app.post("/api/admin/models/<model_id>/toggle")
def admin_toggle_model(model_id: str):
    if not require_admin():
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    found = None
    for m in MODELS:
        if m.get("id") == model_id:
            m["enabled"] = not bool(m.get("enabled", True))
            found = m
            break
    if not found:
        return jsonify({"ok": False, "error": "Model not found"}), 404
    save_models(MODELS)
    return jsonify({
        "ok": True,
        "message": "Model toggled",
        "model": {k: v for k, v in found.items() if k != "api_key"},
    })

@app.post("/api/admin/test")
def admin_test_model():
    if not require_admin():
        return jsonify({"ok": False, "error": "Unauthorized"}), 401
    if not request.is_json:
        return jsonify({"ok": False, "error": "JSON body required"}), 400

    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "Hello")).strip() or "Hello"
    preferred_model = data.get("model")
    chat_id = str(data.get("chat_id") or "admin-test")
    history = CHAT_STATE.get(chat_id, [])
    messages = normalize_messages(message, history=history)

    candidates = pick_candidates(preferred_model=preferred_model)
    if not candidates:
        return jsonify({"ok": False, "error": "No enabled models available"}), 503

    errors = []
    for candidate in candidates:
        api_key = (candidate.get("api_key") or "").strip()
        model = candidate.get("model")
        if not api_key:
            errors.append(f"{candidate.get('name')} missing api_key")
            continue
        try:
            answer = groq_chat(api_key=api_key, model=model, messages=messages)
            return jsonify({
                "ok": True,
                "model_used": model,
                "provider_name": candidate.get("name"),
                "answer": answer,
            })
        except Exception as e:
            errors.append(f"{model}: {e}")
            continue

    return jsonify({"ok": False, "error": "All models failed", "details": errors}), 502

# -----------------------
# Convenience
# -----------------------
@app.errorhandler(404)
def not_found(_):
    return jsonify({"ok": False, "error": "Not found"}), 404

@app.errorhandler(500)
def server_error(err):
    return jsonify({"ok": False, "error": str(err)}), 500

if __name__ == "__main__":
    print(f"Starting {APP_NAME} on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
