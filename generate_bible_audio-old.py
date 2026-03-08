"""
generate_bible_audio.py
=======================
Genera audio TTS para cada versículo de la Biblia NVI,
sube cada archivo a MinIO y registra la referencia en PostgreSQL.

Nomenclatura de archivos: NVI_{book}_{chapter}_{verse:03d}.mp3
Ejemplo: NVI_Gen_1_001.mp3, NVI_Jn_3_016.mp3

Uso:
    python generate_bible_audio.py                        # todos los versículos
    python generate_bible_audio.py --book Gen             # solo un libro
    python generate_bible_audio.py --book Gen --chapter 1 # solo un capítulo
    python generate_bible_audio.py --resume               # salta los ya generados

Variables de entorno requeridas (mismas que server.py):
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY
    MINIO_BUCKET, MINIO_PUBLIC_URL
    POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_DB
    VOICE_FILE  (ruta al archivo .wav de voz, default: /app/voices/default.wav)
    SQLITE_PATH (ruta al .sqlite3, default: /app/nvi.sqlite3)
"""

import os
import re
import uuid
import sqlite3
import tempfile
import argparse
import time

import boto3
from sqlalchemy import create_engine, text
from TTS.api import TTS

# ======================
# CONFIG
# ======================

SQLITE_PATH  = os.getenv("SQLITE_PATH",  "/app/nvi.sqlite3")
VOICE_FILE   = os.getenv("VOICE_FILE",   "/app/voices/default.wav")
LANGUAGE     = os.getenv("TTS_LANGUAGE", "es")
BIBLE_NAME   = "NVI"

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET           = os.getenv("MINIO_BUCKET")
PUBLIC_URL       = os.getenv("MINIO_PUBLIC_URL")

DB_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"
)

# ======================
# INIT CLIENTS
# ======================

print("Connecting to PostgreSQL...")
engine = create_engine(DB_URL)

print("Connecting to MinIO...")
s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

print("Loading XTTS model...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
print("XTTS ready.")

# ======================
# DB: ensure table exists
# ======================

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tts_audio (
                id         UUID PRIMARY KEY,
                bible      TEXT,
                book       TEXT,
                chapter    INTEGER,
                verse      INTEGER,
                voice      TEXT,
                language   TEXT,
                hash       TEXT UNIQUE,
                audio_url  TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
    print("DB ready.")

# ======================
# HELPERS
# ======================

def clean_text(text: str) -> str:
    """Remove newlines and extra spaces from verse text."""
    return re.sub(r'\s+', ' ', text.replace('\n', ' ')).strip()


def decode_verse(raw: float):
    """
    Decode SQLite verse format: 1.016 -> chapter=1, verse=16
    """
    chapter = int(raw)
    verse   = round((raw - chapter) * 1000)
    return chapter, verse


def object_key(book: str, chapter: int, verse: int) -> str:
    """MinIO object name: NVI_Gen_1_016.mp3"""
    return f"{BIBLE_NAME}_{book}_{chapter}_{verse:03d}.mp3"


def already_generated(conn, book: str, chapter: int, verse: int) -> bool:
    row = conn.execute(
        text("SELECT 1 FROM tts_audio WHERE bible=:b AND book=:bo AND chapter=:c AND verse=:v"),
        {"b": BIBLE_NAME, "bo": book, "c": chapter, "v": verse}
    ).fetchone()
    return row is not None


def generate_and_upload(book: str, chapter: int, verse: int, verse_text: str) -> str:
    """Generate TTS audio (WAV), convert to MP3 via ffmpeg, upload to MinIO."""
    import subprocess
    key      = object_key(book, chapter, verse)
    wav_path = None
    mp3_path = None

    try:
        # Step 1: Generate WAV (XTTS native output)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        tts.tts_to_file(
            text=verse_text,
            speaker_wav=VOICE_FILE,
            language=LANGUAGE,
            file_path=wav_path,
        )

        # Step 2: Convert WAV -> MP3 with ffmpeg
        mp3_path = wav_path.replace(".wav", ".mp3")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr}")

        # Step 3: Upload MP3 to MinIO
        s3.upload_file(mp3_path, BUCKET, key, ExtraArgs={"ContentType": "audio/mpeg"})

    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        if mp3_path and os.path.exists(mp3_path):
            os.remove(mp3_path)

    return f"{PUBLIC_URL}/{key}"


def save_to_db(conn, book: str, chapter: int, verse: int, url: str):
    conn.execute(
        text("""
            INSERT INTO tts_audio
                (id, bible, book, chapter, verse, voice, language, hash, audio_url)
            VALUES
                (:id, :bible, :book, :chapter, :verse, :voice, :language, :hash, :url)
            ON CONFLICT (hash) DO NOTHING
        """),
        {
            "id":      uuid.uuid4(),
            "bible":   BIBLE_NAME,
            "book":    book,
            "chapter": chapter,
            "verse":   verse,
            "voice":   os.path.basename(VOICE_FILE),
            "language": LANGUAGE,
            "hash":    object_key(book, chapter, verse),  # unique key
            "url":     url,
        }
    )

# ======================
# MAIN
# ======================

def main():
    parser = argparse.ArgumentParser(description="Generate Bible TTS audio")
    parser.add_argument("--book",    help="Filter by book OSIS code (e.g. Gen, Jn, Rev)")
    parser.add_argument("--chapter", type=int, help="Filter by chapter number")
    parser.add_argument("--resume",  action="store_true", help="Skip already generated verses")
    args = parser.parse_args()

    init_db()

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    cursor      = sqlite_conn.cursor()

    # Build query
    query  = "SELECT book, verse, unformatted FROM verses"
    params = []
    conditions = []

    if args.book:
        conditions.append("book = ?")
        params.append(args.book)
    if args.chapter:
        conditions.append("CAST(verse AS INT) = ?")
        params.append(args.chapter)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY book, verse"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    sqlite_conn.close()

    total   = len(rows)
    done    = 0
    skipped = 0
    errors  = 0

    print(f"\nTotal verses to process: {total}")
    print(f"Voice: {VOICE_FILE}")
    print(f"Bucket: {BUCKET}")
    print("-" * 50)

    for book, raw_verse, unformatted in rows:
        chapter, verse_num = decode_verse(raw_verse)
        verse_text = clean_text(unformatted)

        label = f"{BIBLE_NAME} {book} {chapter}:{verse_num:03d}"

        try:
            with engine.begin() as conn:
                if args.resume and already_generated(conn, book, chapter, verse_num):
                    skipped += 1
                    print(f"  SKIP  {label}")
                    continue

                url = generate_and_upload(book, chapter, verse_num, verse_text)
                save_to_db(conn, book, chapter, verse_num, url)

            done += 1
            print(f"  OK    {label}  ->  {url}")

        except Exception as e:
            errors += 1
            print(f"  ERROR {label}: {e}")
            time.sleep(1)  # brief pause on error

    print("-" * 50)
    print(f"Done: {done} | Skipped: {skipped} | Errors: {errors} | Total: {total}")


if __name__ == "__main__":
    main()
