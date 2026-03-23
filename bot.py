import os
from pathlib import Path

import httpx
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters


def load_local_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


load_local_env()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000/predict")

if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_message or not update.effective_message.text:
        return
    if not update.effective_user or update.effective_user.is_bot:
        return

    text = update.effective_message.text

    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.post(BACKEND_URL, json={"text": text})
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return  # optionally log this

    confidence = data.get("confidence")
    confidence_line = (
        f"Confidence: {confidence:.3f}"
        if isinstance(confidence, (float, int))
        else "Confidence: unavailable"
    )

    response_lines = [
        f"Label: {data.get('label', 'unknown')}",
        f"Intent: {data.get('intent', 'unknown')}",
        f"Manipulation Type: {data.get('manipulation_type', 'unknown')}",
        f"Domain: {data.get('domain', 'unknown')}",
        f"Severity: {data.get('severity', 'unknown')}",
        confidence_line,
    ]
    await update.effective_message.reply_text("\n".join(response_lines))


def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()
