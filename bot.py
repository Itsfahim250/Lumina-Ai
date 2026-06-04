import os
import json
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException
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
MODELS = ["llama-3.1-8b-instant", "gemma2-9b-it"]

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

# FastAPI এর জন্য Pydantic মডেল
class ChatRequest(BaseModel):
    prompt: str

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
        
        keyboard = []
        for i, key in enumerate(API_KEYS):
            # Key এর কিছু অংশ হাইড করে দেখানো (সিকিউরিটির জন্য)
            masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else key
            keyboard.append([InlineKeyboardButton(f"Delete: {masked}", callback_data=f"del_{i}")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("নিচের বাটনে ক্লিক করে API Key ডিলিট করুন:", reply_markup=reply_markup)
        return

    if text == "Project settings":
        # Render-এ হোস্ট করলে RENDER_EXTERNAL_URL অটোমেটিকভাবে সেট হয়
        base_url = os.getenv("RENDER_EXTERNAL_URL", "http://127.0.0.1:8000")
        
        msg = (
            f"⚙️ **Project Settings & API Endpoints**\n\n"
            f"**Base URL:**\n`{base_url}`\n\n"
            f"**Supported Models:**\n"
            f"1. `llama-3.1-8b-instant`\n"
            f"2. `gemma2-9b-it`\n\n"
            f"🔹 **GET Request Example:**\n"
            f"`{base_url}/api/llama-3.1-8b-instant/chat?prompt=Hi`\n\n"
            f"🔹 **POST Request Example:**\n"
            f"URL: `{base_url}/api/gemma2-9b-it/chat`\n"
            f"Body (JSON): `{{\"prompt\": \"Hello AI\"}}`"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
        return

    # যদি ইউজার API Key ইনপুট দেয়
    if USER_STATES.get(chat_id) == "AWAITING_KEY":
        new_key = text.strip()
        if new_key not in API_KEYS:
            API_KEYS.append(new_key)
            save_keys(API_KEYS)
            await update.message.reply_text("✅ API Key সফলভাবে অ্যাড করা হয়েছে!")
        else:
            await update.message.reply_text("⚠️ এই Key টি আগে থেকেই অ্যাড করা আছে।")
        
        USER_STATES[chat_id] = None # স্টেট ক্লিয়ার করা
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
            masked = f"{deleted[:8]}...{deleted[-4:]}" if len(deleted) > 12 else deleted
            await query.edit_message_text(text=f"✅ API Key ডিলিট করা হয়েছে:\n{masked}")
        else:
            await query.edit_message_text(text="⚠️ Key খুঁজে পাওয়া যায়নি বা আগেই ডিলিট হয়েছে।")

# টেলিগ্রাম বট সেটআপ
bot_app = Application.builder().token(BOT_TOKEN).build()
bot_app.add_handler(CommandHandler("start", start))
bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
bot_app.add_handler(CallbackQueryHandler(button_callback))

# ==========================================
# FASTAPI LOGIC (Backend Server)
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI সার্ভার অন হওয়ার সাথে টেলিগ্রাম বট চালু হবে
    await bot_app.initialize()
    await bot_app.start()
    await bot_app.updater.start_polling()
    yield
    # FastAPI সার্ভার বন্ধ হওয়ার সময় টেলিগ্রাম বট অফ হবে
    await bot_app.updater.stop()
    await bot_app.stop()
    await bot_app.shutdown()

app = FastAPI(lifespan=lifespan)

# ফ্রন্টএন্ড থেকে API কল করার জন্য CORS অ্যাড করা হলো
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def fetch_groq_response(model: str, prompt: str):
    if not API_KEYS:
        return {"error": "No API keys configured on backend. Add keys via Telegram Bot."}

    if model not in MODELS:
        return {"error": f"Invalid model. Supported models are: {', '.join(MODELS)}"}

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # যতগুলো API Key আছে, সবগুলোর উপর লুপ চলবে
        for key in API_KEYS:
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            try:
                async with session.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {"response": data["choices"][0]["message"]["content"]}
                    else:
                        # লিমিট শেষ বা এরর আসলে পরের Key ট্রাই করবে
                        continue
            except Exception:
                # নেটওয়ার্ক এরর হলেও পরের Key ট্রাই করবে
                continue
    
    # যদি কোনো Key দিয়েই কাজ না হয়
    return {"error": "All API keys failed or rate limits exceeded."}

@app.get("/api/{model_name}/chat")
async def chat_get(model_name: str, prompt: str):
    res = await fetch_groq_response(model_name, prompt)
    if "error" in res:
        raise HTTPException(status_code=500, detail=res["error"])
    return res

@app.post("/api/{model_name}/chat")
async def chat_post(model_name: str, req: ChatRequest):
    res = await fetch_groq_response(model_name, req.prompt)
    if "error" in res:
        raise HTTPException(status_code=500, detail=res["error"])
    return res

@app.get("/")
async def root():
    return {"message": "Backend is running. Open your Telegram Bot to manage Settings."}

if __name__ == "__main__":
    import uvicorn
    # Render ডিফল্টভাবে PORT environment variable প্রোভাইড করে
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
