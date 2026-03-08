import os
import uuid
import hashlib
import tempfile
import logging

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from TTS.api import TTS

import boto3
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("biblia-ai")

app = FastAPI()

# ======================
# CONFIG
# ======================

VOICE_DIR    = "/app/voices"
OUTPUT_DIR   = "/app/outputs"
TEMPLATE_DIR = "/app/templates"

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMPLATE_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# ======================
# LOAD MODEL
# ======================

print("Loading XTTS...")
tts = TTS(MODEL)
print("XTTS ready")

# ======================
# MINIO
# ======================

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("MINIO_ENDPOINT"),
    aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("MINIO_SECRET_KEY")
)

BUCKET     = os.getenv("MINIO_BUCKET")
PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL")

def get_audio_url(object_name: str) -> str:
    use_presigned = os.getenv("USE_PRESIGNED_URLS", "false").lower() == "true"
    if use_presigned:
        return s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": BUCKET, "Key": object_name},
            ExpiresIn=3600
        )
    return f"{PUBLIC_URL}/{object_name}"

# ======================
# DATABASE
# ======================

POSTGRES_USER     = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST     = os.getenv("POSTGRES_HOST")
POSTGRES_DB       = os.getenv("POSTGRES_DB")

logger.info(f"DB CONFIG -> user={POSTGRES_USER} host={POSTGRES_HOST} db={POSTGRES_DB}")

DB_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"
engine = create_engine(DB_URL)

def init_db():
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tts_audio (
                    id         UUID PRIMARY KEY,
                    bible      VARCHAR,
                    book       VARCHAR,
                    chapter    INTEGER,
                    verse      INTEGER,
                    voice      VARCHAR,
                    language   VARCHAR,
                    hash       VARCHAR UNIQUE,
                    audio_url  TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS chat_log (
                    id         UUID PRIMARY KEY,
                    message    TEXT,
                    response   TEXT,
                    voice      VARCHAR,
                    language   VARCHAR,
                    audio_url  TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS reading_log (
                    id         UUID PRIMARY KEY,
                    bible      VARCHAR,
                    voice      VARCHAR,
                    language   VARCHAR,
                    full_text  TEXT,
                    audio_url  TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """))
        logger.info("DB tables ready")
    except Exception as e:
        logger.error(f"DB init error: {e}")

init_db()

# ======================
# MODELS
# ======================

class VerseRequest(BaseModel):
    bible:    str
    book:     str
    chapter:  int
    verse:    int
    voice:    str
    language: str
    text:     str

class ChatRequest(BaseModel):
    message:  str
    voice:    str
    language: str = "es"

class DynamicReading(BaseModel):
    bible:    str
    voice:    str
    language: str
    verses:   list

# ======================
# HASH
# ======================

def generate_hash(req):
    key = "|".join([
        req.bible, req.book,
        str(req.chapter), str(req.verse),
        req.voice, req.language, req.text
    ])
    return hashlib.sha256(key.encode()).hexdigest()


# ======================
# CHUNK TEXT (XTTS limit ~250 tokens / ~900 chars)
# ======================

import re

def split_text(text: str, max_chars: int = 200) -> list:
    """
    Splits text into chunks respecting sentence boundaries.
    XTTS v2 crashes with texts longer than ~250 tokens.
    """
    sentences = re.split(r'(?<=[.!?;,]) +', text.strip())
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks if chunks else [text]


def tts_chunks_to_file(text: str, voice_path: str, language: str, output_file: str):
    """
    Generates audio splitting text in chunks and concatenating with pydub.
    """
    import pydub
    chunks = split_text(text)
    logger.info(f"TTS chunks: {len(chunks)} for text length {len(text)}")
    if len(chunks) == 1:
        tts.tts_to_file(text=chunks[0], speaker_wav=voice_path, language=language, file_path=output_file)
        return
    segments = []
    tmp_files = []
    for i, chunk in enumerate(chunks):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        tts.tts_to_file(text=chunk, speaker_wav=voice_path, language=language, file_path=tmp.name)
        segments.append(pydub.AudioSegment.from_wav(tmp.name))
        tmp_files.append(tmp.name)
    combined = segments[0]
    for seg in segments[1:]:
        combined += seg
    combined.export(output_file, format="mp3")
    for f in tmp_files:
        os.remove(f)

# ======================
# HOME
# ======================

@app.get("/", response_class=HTMLResponse)
def home():
    template_path = os.path.join(TEMPLATE_DIR, "index.html")
    with open(template_path) as f:
        return f.read()

# ======================
# DEBUG DB
# ======================

@app.get("/db-check")
def db_check():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM tts_audio")).fetchone()
            chats  = conn.execute(text("SELECT COUNT(*) FROM chat_log")).fetchone()
            reads  = conn.execute(text("SELECT COUNT(*) FROM reading_log")).fetchone()
        return {
            "status": "ok",
            "tts_audio_count": result[0],
            "chat_log_count":  chats[0],
            "reading_log_count": reads[0]
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# ======================
# LIST VOICES
# ======================

@app.get("/voices")
def voices():
    return os.listdir(VOICE_DIR)

# ======================
# STREAM
# ======================

@app.get("/stream")
def stream(text: str, voice: str = "default", language: str = "es"):
    voice_path  = f"{VOICE_DIR}/{voice}"
    file_id     = str(uuid.uuid4())
    output_file = f"{OUTPUT_DIR}/{file_id}.mp3"

    tts_chunks_to_file(text=text, voice_path=voice_path, language=language, output_file=output_file)

    return JSONResponse({"audio_file": f"/outputs/{file_id}.mp3"})

# ======================
# VERSE AUDIO (cache)
# ======================

@app.post("/verse-audio")
def generate(req: VerseRequest):
    h = generate_hash(req)

    try:
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT audio_url FROM tts_audio WHERE hash=:h"),
                {"h": h}
            ).fetchone()
            if row:
                logger.info(f"Cache hit: {h}")
                return {"audio_url": row[0], "cached": True}
    except Exception as e:
        logger.error(f"DB read error: {e}")

    voice_path = f"{VOICE_DIR}/{req.voice}"

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_name = tmp.name

    tts_chunks_to_file(text=req.text, voice_path=voice_path, language=req.language, output_file=tmp_name)

    object_name = f"{h}.mp3"
    s3.upload_file(tmp_name, BUCKET, object_name)
    os.remove(tmp_name)

    url = get_audio_url(object_name)

    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO tts_audio
                    (id,bible,book,chapter,verse,voice,language,hash,audio_url)
                    VALUES (:id,:b,:bo,:c,:v,:vo,:l,:h,:u)
                    ON CONFLICT (hash) DO NOTHING
                """),
                {
                    "id": uuid.uuid4(),
                    "b":  req.bible,
                    "bo": req.book,
                    "c":  req.chapter,
                    "v":  req.verse,
                    "vo": req.voice,
                    "l":  req.language,
                    "h":  h,
                    "u":  url
                }
            )
        logger.info(f"DB insert OK: {h}")
    except Exception as e:
        logger.error(f"DB insert error: {e}")

    return {"audio_url": url, "cached": False}

# ======================
# GENERATE CHAPTER
# ======================

@app.post("/generate-chapter")
def generate_chapter(data: dict):
    results = []
    for verse in data["verses"]:
        req = VerseRequest(
            bible=data["bible"], book=data["book"],
            chapter=data["chapter"], verse=verse["verse"],
            voice=data["voice"], language=data["language"],
            text=verse["text"]
        )
        results.append(generate(req))
    return results

# ======================
# VOICE CLONE
# ======================

@app.post("/clone")
async def clone(file: UploadFile):
    path = f"{VOICE_DIR}/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"status": "voice added"}

# ======================
# AI CHAT BIBLICO
# ======================

@app.post("/ai-chat")
def ai_chat(req: ChatRequest):
    response_text = f"Respuesta biblica a: {req.message}"
    voice_path    = f"{VOICE_DIR}/{req.voice}"

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_name = tmp.name

    tts_chunks_to_file(text=response_text, voice_path=voice_path, language=req.language, output_file=tmp_name)

    object_name = f"chat_{uuid.uuid4()}.mp3"
    s3.upload_file(tmp_name, BUCKET, object_name)
    os.remove(tmp_name)

    url = get_audio_url(object_name)

    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO chat_log (id,message,response,voice,language,audio_url)
                    VALUES (:id,:msg,:resp,:vo,:lang,:url)
                """),
                {
                    "id":   uuid.uuid4(),
                    "msg":  req.message,
                    "resp": response_text,
                    "vo":   req.voice,
                    "lang": req.language,
                    "url":  url
                }
            )
        logger.info("chat_log insert OK")
    except Exception as e:
        logger.error(f"chat_log insert error: {e}")

    return {"text": response_text, "audio_url": url}

# ======================
# DYNAMIC READING
# ======================

@app.post("/dynamic-reading")
def dynamic_reading(data: DynamicReading):
    full_text  = " ".join(data.verses)
    voice_path = f"{VOICE_DIR}/{data.voice}"

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_name = tmp.name

    tts_chunks_to_file(text=full_text, voice_path=voice_path, language=data.language, output_file=tmp_name)

    object_name = f"reading_{uuid.uuid4()}.mp3"
    s3.upload_file(tmp_name, BUCKET, object_name)
    os.remove(tmp_name)

    url = get_audio_url(object_name)

    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO reading_log (id,bible,voice,language,full_text,audio_url)
                    VALUES (:id,:bible,:vo,:lang,:txt,:url)
                """),
                {
                    "id":    uuid.uuid4(),
                    "bible": data.bible,
                    "vo":    data.voice,
                    "lang":  data.language,
                    "txt":   full_text,
                    "url":   url
                }
            )
        logger.info("reading_log insert OK")
    except Exception as e:
        logger.error(f"reading_log insert error: {e}")

    return {"audio_url": url}