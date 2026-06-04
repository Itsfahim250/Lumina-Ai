import os
import json
import asyncio
import aiohttp
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    KeyboardButton
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# .env ফাইল লোড করা
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

KEYS_FILE = "api_keys.json"
# মডেল ম্যাপিং (আপনার রিকোয়েস্ট অনুযায়ী)
MODEL_MAPPING = {
    "llama-3-8b": "llama-3.1-8b-instant",
    "llama-3-70b": "llama-3.1-70b-versatile",
    "mixtral-8x7b": "mixtral-8x7b-32768",
    "gemma2-9b": "gemma2-9b-it"
}

# চ্যাট হিস্ট্রি স্টোর করার জন্য গ্লোবাল ডিকশনারি
CHAT_HISTORY = {}

def load_keys():
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_keys(keys):
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f)

API_KEYS = load_keys()
USER_STATES = {}

class ChatRequest(BaseModel):
    prompt: str

# ==========================================
# TELEGRAM BOT LOGIC
# ==========================================

def get_main_keyboard():
    keyboard = [
        [KeyboardButton("Add model"), KeyboardButton("Delete model")],
        [KeyboardButton("Project settings")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "স্বাগতম! এটি আপনার Backend Admin Panel.\nনিচের বাটনগুলো ব্যবহার করে API Key ম্যানেজ করুন:",
        reply_markup=get_main_keyboard()
    )

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    chat_id = update.effective_chat.id

    if text == "Add model":
        USER_STATES[chat_id] = "AWAITING_KEY"
        await update.message.reply_text("অনুগ্রহ করে নতুন Groq API Key দিন:")
        return

    if text == "Delete model":
        if not API_KEYS:
            await update.message.reply_text("আপনার কোনো API Key সেভ করা নেই।")
            return
        keyboard = [[InlineKeyboardButton(f"Delete: {k[:8]}...", callback_data=f"del_{i}")] for i, k in enumerate(API_KEYS)]
        await update.message.reply_text("নিচের বাটনে ক্লিক করে API Key ডিলিট করুন:", reply_markup=InlineKeyboardMarkup(keyboard))
        return

    if text == "Project settings":
        base_url = os.getenv("RENDER_EXTERNAL_URL", "http://127.0.0.1:8000")
        msg = (
            f"⚙️ **Project Settings & API Guide**\n\n"
            f"**Base URL:** `{base_url}`\n\n"
            f"**Supported Models:**\n"
            f"1. `llama-3-8b` | 2. `llama-3-70b`\n"
            f"3. `mixtral-8x7b` | 4. `gemma2-9b`\n\n"
            f"🔹 **Chat (GET):**\n"
            f"`{base_url}/api/llama-3-8b/chat?prompt=Hi&session_id=user1`\n\n"
            f"🔹 **Voice to Text (POST):**\n"
            f"Endpoint: `/api/transcribe`\n"
            f"Field: `file` (Audio file)"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
        return

    if USER_STATES.get(chat_id) == "AWAITING_KEY":
        new_key = text.strip()
        if new_key not in API_KEYS:
            API_KEYS.append(new_key)
            save_keys(API_KEYS)
            await update.message.reply_text("✅ API Key সফলভাবে অ্যাড করা হয়েছে!")
        USER_STATES[chat_id] = None
        return

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data.startswith("del_"):
        idx = int(query.data.split("_")[1])
        if 0 <= idx < len(API_KEYS):
            API_KEYS.pop(idx)
            save_keys(API_KEYS)
            await query.edit_message_text(text="✅ API Key ডিলিট করা হয়েছে।")

bot_app = Application.builder().token(BOT_TOKEN).build()
bot_app.add_handler(CommandHandler("start", start))
bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
bot_app.add_handler(CallbackQueryHandler(button_callback))

# ==========================================
# FASTAPI LOGIC
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await bot_app.initialize()
    await bot_app.start()
    await bot_app.updater.start_polling()
    yield
    await bot_app.updater.stop()
    await bot_app.stop()
    await bot_app.shutdown()

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

async def fetch_groq_chat(model_alias: str, prompt: str, session_id: str):
    if not API_KEYS: return {"error": "No API Keys found"}
    
    model_id = MODEL_MAPPING.get(model_alias)
    if not model_id: return {"error": "Invalid Model Name"}

    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = []
    
    CHAT_HISTORY[session_id].append({"role": "user", "content": prompt})
    messages = CHAT_HISTORY[session_id][-10:] # শেষ ১০টি মেসেজ মেমোরি হিসেবে থাকবে

    async with aiohttp.ClientSession() as session:
        for key in API_KEYS:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {"model": model_id, "messages": messages, "temperature": 0.7}
            try:
                async with session.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        ans = data["choices"][0]["message"]["content"]
                        CHAT_HISTORY[session_id].append({"role": "assistant", "content": ans})
                        return {"response": ans, "session_id": session_id}
            except: continue
    return {"error": "All keys failed or Rate limit reached"}

@app.get("/api/{model_name}/chat")
async def chat_get(model_name: str, prompt: str, session_id: str = "default"):
    res = await fetch_groq_chat(model_name, prompt, session_id)
    if "error" in res: raise HTTPException(status_code=500, detail=res["error"])
    return res

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not API_KEYS: raise HTTPException(status_code=500, detail="No Keys")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    async with aiohttp.ClientSession() as session:
        for key in API_KEYS:
            data = aiohttp.FormData()
            data.add_field('file', open(tmp_path, 'rb'))
            data.add_field('model', 'whisper-large-v3')
            headers = {"Authorization": f"Bearer {key}"}
            try:
                async with session.post("https://api.groq.com/openai/v1/audio/transcriptions", headers=headers, data=data) as resp:
                    if resp.status == 200:
                        res_data = await resp.json()
                        os.remove(tmp_path)
                        return res_data
            except: continue
    
    if os.path.exists(tmp_path): os.remove(tmp_path)
    raise HTTPException(status_code=500, detail="Transcription failed")

@app.get("/")
async def root():
    return {"message": "Backend is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
