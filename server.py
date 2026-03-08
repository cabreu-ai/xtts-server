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
    Splits text into chunks, generates WAV per chunk, concatenates
    raw PCM bytes and writes final WAV. Works without pydub.
    output_file can be .wav or .mp3 - XTTS writes WAV natively.
    """
    import numpy as np
    import scipy.io.wavfile as wavfile

    chunks = split_text(text)
    logger.info(f"TTS chunks: {len(chunks)} for text length {len(text)}")

    # Always write to a .wav temp path first, then rename/move to output_file
    wav_output = output_file if output_file.endswith(".wav") else output_file.replace(".mp3", ".wav")

    if len(chunks) == 1:
        tts.tts_to_file(text=chunks[0], speaker_wav=voice_path, language=language, file_path=wav_output)
        if wav_output != output_file:
            os.rename(wav_output, output_file)
        return

    tmp_files = []
    arrays = []
    sample_rate = None

    for chunk in chunks:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        tts.tts_to_file(text=chunk, speaker_wav=voice_path, language=language, file_path=tmp.name)
        sr, data = wavfile.read(tmp.name)
        sample_rate = sr
        arrays.append(data)
        tmp_files.append(tmp.name)

    combined = np.concatenate(arrays)
    wavfile.write(wav_output, sample_rate, combined)

    if wav_output != output_file:
        os.rename(wav_output, output_file)

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
    results = {}
    for table in ["tts_audio", "chat_log", "reading_log"]:
        try:
            with engine.connect() as conn:
                n = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                results[table] = {"count": n[0], "status": "ok"}
        except Exception as e:
            results[table] = {"status": "error", "detail": str(e)}
    return results

@app.get("/db-test-insert")
def db_test_insert():
    """Test endpoint to verify DB inserts work correctly."""
    test_id = uuid.uuid4()
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO tts_audio (id,bible,book,chapter,verse,voice,language,hash,audio_url)
                VALUES (:id,:b,:bo,:c,:v,:vo,:l,:h,:u)
                ON CONFLICT (hash) DO NOTHING
            """), {
                "id": test_id, "b": "TEST", "bo": "Test",
                "c": 1, "v": 1, "vo": "default", "l": "es",
                "h": f"test_{test_id}", "u": "https://test.url/audio.wav"
            })
        return {"status": "ok", "inserted_id": str(test_id)}
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

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name

    tts_chunks_to_file(text=req.text, voice_path=voice_path, language=req.language, output_file=tmp_name)

    object_name = f"{h}.wav"
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


@app.get("/debug-verse")
def debug_verse(text: str = "Dios es amor", voice: str = "default", language: str = "es"):
    """Step by step debug: TTS -> MinIO -> DB"""
    import traceback
    result = {"step_tts": None, "step_minio": None, "step_db": None}

    # STEP 1: TTS
    try:
        voice_path = f"{VOICE_DIR}/{voice}"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_name = tmp.name
        tts_chunks_to_file(text=text, voice_path=voice_path, language=language, output_file=tmp_name)
        size = os.path.getsize(tmp_name)
        result["step_tts"] = f"OK - file size: {size} bytes at {tmp_name}"
    except Exception as e:
        result["step_tts"] = f"ERROR: {traceback.format_exc()}"
        return result

    # STEP 2: MinIO upload
    try:
        object_name = f"debug_test_{uuid.uuid4()}.wav"
        s3.upload_file(tmp_name, BUCKET, object_name)
        url = get_audio_url(object_name)
        result["step_minio"] = f"OK - url: {url}"
        os.remove(tmp_name)
    except Exception as e:
        result["step_minio"] = f"ERROR: {traceback.format_exc()}"
        return result

    # STEP 3: DB insert
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO tts_audio (id,bible,book,chapter,verse,voice,language,hash,audio_url)
                VALUES (:id,:b,:bo,:c,:v,:vo,:l,:h,:u)
            """), {
                "id": uuid.uuid4(), "b": "DEBUG", "bo": "Debug",
                "c": 0, "v": 0, "vo": voice, "l": language,
                "h": f"debug_{uuid.uuid4()}", "u": url
            })
        result["step_db"] = "OK - inserted"
    except Exception as e:
        result["step_db"] = f"ERROR: {traceback.format_exc()}"

    return result

# ======================
# AI CHAT BIBLICO
# ======================

@app.post("/ai-chat")
def ai_chat(req: ChatRequest):
    response_text = f"Respuesta biblica a: {req.message}"
    voice_path    = f"{VOICE_DIR}/{req.voice}"

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name

    tts_chunks_to_file(text=response_text, voice_path=voice_path, language=req.language, output_file=tmp_name)

    object_name = f"chat_{uuid.uuid4()}.wav"
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

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_name = tmp.name

    tts_chunks_to_file(text=full_text, voice_path=voice_path, language=data.language, output_file=tmp_name)

    object_name = f"reading_{uuid.uuid4()}.wav"
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