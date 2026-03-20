#!/usr/bin/env python3
"""
brain_bot.py - The Brain: Telegram Bot
Receives text messages and voice notes from Telegram,
processes them with Claude, and writes structured notes
to Dropbox via the Dropbox API.
"""

import os
import json
import re
import requests
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config from environment variables ─────────────────────────────────────────
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
DROPBOX_TOKEN = os.environ["DROPBOX_TOKEN"]
ALLOWED_USER = os.environ.get("ALLOWED_TELEGRAM_USER", "")  # Your Telegram username
MODEL = "claude-sonnet-4-6"

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
DROPBOX_API = "https://content.dropboxapi.com/2/files/upload"
VAULT_BASE = "/The Brain"

FOLDERS = {
    "inbox":      "00 - Inbox",
    "open_loops": "01 - Open Loops",
    "projects":   "02 - Projects",
    "people":     "03 - People",
    "meetings":   "04 - Meetings",
    "decisions":  "05 - Decisions & Context",
    "tasks":      "06 - Google Tasks Sync",
    "journal":    "07 - Journal",
}

SYSTEM_PROMPT = """You are Courtney's second brain assistant. Courtney is the Director of Finance & HR at Hudson Restoration - a solo function covering HR, payroll, accounting, legal, and admin. Key people: Nick (President), Steve (owner), Pam (direct report), Sarah, Siobhan, Greg, Tracy, Tina.

Your job is to receive a raw brain dump and return a structured JSON note.

Return ONLY valid JSON with this exact structure:
{
  "folder": "one of: inbox, open_loops, projects, people, meetings, decisions, journal",
  "title": "concise note title (no date needed)",
  "tags": ["tag1", "tag2"],
  "summary": "1-2 sentence plain english summary of what this is about",
  "content": "full structured markdown content for the note body",
  "next_actions": ["action 1", "action 2"],
  "people_mentioned": ["Name1", "Name2"],
  "urgency": "high | medium | low"
}

Folder routing guide:
- inbox: unclear, mixed, or quick captures that need processing later
- open_loops: something in progress with no resolution yet
- projects: active initiative with a goal and timeline
- people: notes about a specific person (use their name as title)
- meetings: prep or notes for a specific meeting
- decisions: something decided, or context Courtney needs to remember
- journal: personal reflection, end of day, emotional processing

For the content field, write clean markdown. Use headers, bullets, and bold where helpful.
If a brain dump contains multiple items (tasks for a person, multiple projects, etc),
break them out clearly in the content with separate sections.
Write as if you're a smart EA capturing this for Courtney to read back later."""


def classify_and_structure(raw_input: str) -> dict:
    """Send raw capture to Claude, get structured note back."""
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": 2000,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"Brain dump to process:\n\n{raw_input}"}],
        },
        timeout=30,
    )
    response.raise_for_status()
    text = response.json()["content"][0]["text"].strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def build_note_content(note: dict) -> str:
    """Build the full markdown note."""
    tags_str = "\n".join(f"  - {t}" for t in note.get("tags", []))
    people_str = "\n".join(f"  - {p}" for p in note.get("people_mentioned", []))
    next_actions_str = "\n".join(f"- [ ] {a}" for a in note.get("next_actions", []))

    frontmatter = f"""---
created: {datetime.now().strftime("%Y-%m-%d %H:%M")}
folder: {note.get("folder", "inbox")}
urgency: {note.get("urgency", "medium")}
tags:
{tags_str if tags_str else "  - capture"}
people:
{people_str if people_str else "  - none"}
---
"""
    body = f"""# {note["title"]}

> {note.get("summary", "")}

{note["content"]}

---
## Next Actions
{next_actions_str if next_actions_str else "- [ ] Review and process"}
"""
    return frontmatter + body


def write_to_dropbox(note: dict, content: str) -> bool:
    """Upload note to Dropbox."""
    folder_key = note.get("folder", "inbox")
    folder_name = FOLDERS.get(folder_key, "00 - Inbox")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    title = note.get("title", "Untitled")
    filename = f"{timestamp} {title}.md"
    dropbox_path = f"{VAULT_BASE}/{folder_name}/{filename}"

    response = requests.post(
        DROPBOX_API,
        headers={
            "Authorization": f"Bearer {DROPBOX_TOKEN}",
            "Content-Type": "application/octet-stream",
            "Dropbox-API-Arg": json.dumps({
                "path": dropbox_path,
                "mode": "add",
                "autorename": True,
                "mute": False,
            }),
        },
        data=content.encode("utf-8"),
        timeout=30,
    )
    response.raise_for_status()
    return True


def transcribe_voice(file_id: str) -> str:
    """Download voice note from Telegram and transcribe using OpenAI Whisper API."""
    # Get file path from Telegram
    file_info = requests.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id}).json()
    file_path = file_info["result"]["file_path"]
    file_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"

    # Download audio
    audio_data = requests.get(file_url).content

    # Send to OpenAI Whisper for transcription
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    if not OPENAI_API_KEY:
        log.error("OPENAI_API_KEY not set - cannot transcribe voice")
        return None

    response = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        files={"file": ("voice.ogg", audio_data, "audio/ogg")},
        data={"model": "whisper-1"},
        timeout=60,
    )

    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        log.error(f"Whisper transcription failed: {response.text}")
        return None


def send_message(chat_id: int, text: str):
    """Send a message back to the user."""
    requests.post(
        f"{TELEGRAM_API}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=10,
    )


def process_update(update: dict):
    """Handle an incoming Telegram update."""
    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    username = message.get("from", {}).get("username", "")

    # Security - only respond to you
    if ALLOWED_USER and username != ALLOWED_USER:
        log.warning(f"Blocked message from @{username}")
        return

    text = message.get("text", "")
    voice = message.get("voice")

    # Handle voice note
    if voice:
        send_message(chat_id, "🎙 Got your voice note, transcribing...")
        transcribed = transcribe_voice(voice["file_id"])
        if not transcribed:
            send_message(chat_id, "Sorry, couldn't transcribe that. Try sending as text.")
            return
        text = transcribed
        send_message(chat_id, f"_Transcribed:_ {text[:200]}{'...' if len(text) > 200 else ''}\n\nFiling...")

    if not text or text.startswith("/"):
        if text == "/start":
            send_message(chat_id, "👋 The Brain is ready. Send me a text or voice note and I'll file it.")
        return

    # Process and file
    try:
        send_message(chat_id, "⚡ Processing...")
        note = classify_and_structure(text)
        content = build_note_content(note)
        write_to_dropbox(note, content)

        urgency_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(note.get("urgency", "medium"), "🟡")
        reply = (
            f"✅ *{note['title']}*\n"
            f"📁 {note['folder'].replace('_', ' ').title()}\n"
            f"{urgency_emoji} {note.get('urgency', 'medium').title()} priority\n"
        )
        if note.get("next_actions"):
            reply += f"📋 {len(note['next_actions'])} action(s) captured"

        send_message(chat_id, reply)

    except Exception as e:
        log.error(f"Error processing message: {e}")
        send_message(chat_id, "Something went wrong filing that. Try again.")


def main():
    """Long-polling loop."""
    log.info("The Brain bot is running...")
    offset = 0

    while True:
        try:
            resp = requests.get(
                f"{TELEGRAM_API}/getUpdates",
                params={"offset": offset, "timeout": 30},
                timeout=35,
            )
            updates = resp.json().get("result", [])
            for update in updates:
                process_update(update)
                offset = update["update_id"] + 1
        except Exception as e:
            log.error(f"Polling error: {e}")
            import time
            time.sleep(5)


if __name__ == "__main__":
    main()
