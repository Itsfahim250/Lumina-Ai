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

# Groq মডেল ম্যাপিং
MODEL_MAPPING = {
    "llama-3-8b": "llama-3.1-8b-instant",
    "llama-3-70b": "llama-3.1-70b-versatile",
    "mixtral-8x7b": "mixtral-8x7b-32768",
    "gemma2-9b": "gemma2-9b-it"
}

# চ্যাট হিস্ট্রি স্টোর করার ডিকশনারি
CHAT_HISTORY = {}

# API Keys সেভ এবং লোড করার ফাংশন
def load_keys():
    if os.path.exists(KEYS_FILE):
        try:
            with open(KEYS_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return []

def save_keys(keys):
    with open(KEYS_FILE, "w") as f:
        json.dump(keys, f)

API_KEYS = load_keys()
USER_STATES = {}

# ==========================================
# TELEGRAM BOT LOGIC (Admin Panel)
# ==========================================

def get_main_keyboard():
    keyboard = [
        [KeyboardButton("Add model"), KeyboardButton("Delete model")],
        [KeyboardButton("Project settings")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Backend Admin Panel-এ স্বাগতম!\nনিচের বাটনগুলো ব্যবহার করে API Key ম্যানেজ করুন:",
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
        
        keyboard = []
        for i, key in enumerate(API_KEYS):
            masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else key
            keyboard.append([InlineKeyboardButton(f"Delete: {masked}", callback_data=f"del_{i}")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("নিচের বাটনে ক্লিক করে API Key ডিলিট করুন:", reply_markup=reply_markup)
        return

    if text == "Project settings":
        base_url = os.getenv("RENDER_EXTERNAL_URL", "http://127.0.0.1:8000")
        
        msg = (
            f"⚙️ **Updated Project Settings & API Guide**\n\n"
            f"**🌐 Base URL:**\n`{base_url}`\n\n"
            f"**🤖 Supported Models:**\n"
            f"1. `llama-3-8b` (LLaMA 3.1 8B)\n"
            f"2. `llama-3-70b` (LLaMA 3.1 70B)\n"
            f"3. `mixtral-8x7b` (Mixtral 8x7B)\n"
            f"4. `gemma2-9b` (Gemma 2 9B)\n\n"
            f"🔹 **Chat API (GET Method with Session):**\n"
            f"URL: `{base_url}/api/llama-3-8b/chat?prompt=Hi&session_id=user123`\n\n"
            f"🔹 **Speech-to-Text (Whisper):**\n"
            f"URL: `{base_url}/api/transcribe` (Method: POST)\n\n"
            f"💡 *টিপস: একই session_id ব্যবহার করলে AI পূর্বের কথা মনে রাখবে।* "
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
        return

    if USER_STATES.get(chat_id) == "AWAITING_KEY":
        new_key = text.strip()
        if new_key not in API_KEYS:
            API_KEYS.append(new_key)
            save_keys(API_KEYS)
            await update.message.reply_text("✅ API Key সফলভাবে অ্যাড করা হয়েছে!")
        else:
            await update.message.reply_text("⚠️ এই Key টি আগে থেকেই অ্যাড করা আছে।")
        USER_STATES[chat_id] = None
        return

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    if data.startswith("del_"):
        idx = int(data.split("_")[1])
        if 0 <= idx < len(API_KEYS):
            deleted = API_KEYS.pop(idx)
            save_keys(API_KEYS)
            await query.edit_message_text(text=f"✅ API Key ডিলিট করা হয়েছে।")

bot_app = Application.builder().token(BOT_TOKEN).build()
bot_app.add_handler(CommandHandler("start", start))
bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
bot_app.add_handler(CallbackQueryHandler(button_callback))

# ==========================================
# FASTAPI LOGIC (Backend Server)
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

async def fetch_groq_response(model_alias: str, prompt: str, session_id: str):
    if not API_KEYS:
        return {"error": "No API keys configured. Add keys via Telegram Bot."}

    model_id = MODEL_MAPPING.get(model_alias)
    if not model_id:
        return {"error": f"Invalid model. Choose: {', '.join(MODEL_MAPPING.keys())}"}

    # সেশন হিস্ট্রি ম্যানেজমেন্ট
    if session_id not in CHAT_HISTORY:
        CHAT_HISTORY[session_id] = []
    
    CHAT_HISTORY[session_id].append({"role": "user", "content": prompt})
    
    # মেমোরি ধরে রাখার জন্য শেষ ১০টি মেসেজ পাঠানো
    messages = CHAT_HISTORY[session_id][-10:]

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.7,
    }

    async with aiohttp.ClientSession() as session:
        for key in API_KEYS:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            try:
                async with session.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        ai_response = data["choices"][0]["message"]["content"]
                        # হিস্ট্রিতে এআই এর উত্তর যোগ করা
                        CHAT_HISTORY[session_id].append({"role": "assistant", "content": ai_response})
                        return {"response": ai_response, "session_id": session_id}
            except Exception:
                continue
    return {"error": "All API keys failed or rate limits exceeded."}

@app.get("/api/{model_name}/chat")
async def chat_get(model_name: str, prompt: str, session_id: str = "default"):
    res = await fetch_groq_response(model_name, prompt, session_id)
    if "error" in res:
        raise HTTPException(status_code=500, detail=res["error"])
    return res

@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not API_KEYS:
        raise HTTPException(status_code=500, detail="No API Keys configured")

    # ফাইলটি সাময়িকভাবে সেভ করা
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
            except:
                continue
    
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    raise HTTPException(status_code=500, detail="Transcription failed on all keys")

@app.get("/")
async def root():
    return {"message": "Backend is running. Use Telegram Bot to manage Settings."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
