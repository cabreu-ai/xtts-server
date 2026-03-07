import os
import uuid
import hashlib
import tempfile

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel

from TTS.api import TTS

import boto3
from sqlalchemy import create_engine, text

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html") as f:
        return f.read()

# ======================
# CONFIG
# ======================

VOICE_DIR = "/app/voices"
OUTPUT_DIR = "/app/outputs"

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

BUCKET = os.getenv("MINIO_BUCKET")
PUBLIC_URL = os.getenv("MINIO_PUBLIC_URL")

# ======================
# DATABASE
# ======================

DB_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"

engine = create_engine(DB_URL)

# ======================
# MODELS
# ======================

class VerseRequest(BaseModel):
    bible:str
    book:str
    chapter:int
    verse:int
    voice:str
    language:str
    text:str


class ChatRequest(BaseModel):
    message:str
    voice:str
    language:str="es"


class DynamicReading(BaseModel):
    bible:str
    voice:str
    language:str
    verses:list


# ======================
# HASH
# ======================

def generate_hash(req):

    key="|".join([
        req.bible,
        req.book,
        str(req.chapter),
        str(req.verse),
        req.voice,
        req.language,
        req.text
    ])

    return hashlib.sha256(key.encode()).hexdigest()


# ======================
# LIST VOICES
# ======================

@app.get("/voices")
def voices():
    return os.listdir(VOICE_DIR)


# ======================
# STREAM (sin cache)
# ======================

@app.get("/stream")
def stream(text:str,voice:str="default",language:str="es"):

    voice_path=f"{VOICE_DIR}/{voice}"

    file_id=str(uuid.uuid4())
    output_file=f"{OUTPUT_DIR}/{file_id}.mp3"

    tts.tts_to_file(
        text=text,
        speaker_wav=voice_path,
        language=language,
        file_path=output_file
    )

    return FileResponse(
        output_file,
        media_type="audio/mpeg",
        filename="audio.mp3"
    )


# ======================
# GENERATE AUDIO (cache)
# ======================

@app.post("/verse-audio")
def generate(req:VerseRequest):

    h=generate_hash(req)

    with engine.begin() as conn:

        row=conn.execute(
            text("SELECT audio_url FROM tts_audio WHERE hash=:h"),
            {"h":h}
        ).fetchone()

        if row:
            return {"audio_url":row[0],"cached":True}

        voice_path=f"{VOICE_DIR}/{req.voice}"

        with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:

            tts.tts_to_file(
                text=req.text,
                speaker_wav=voice_path,
                language=req.language,
                file_path=tmp.name
            )

            object_name=f"{h}.mp3"

            s3.upload_file(tmp.name,BUCKET,object_name)

        url=f"{PUBLIC_URL}/{object_name}"

        conn.execute(
            text("""
                INSERT INTO tts_audio
                (id,bible,book,chapter,verse,voice,language,hash,audio_url)
                VALUES
                (:id,:b,:bo,:c,:v,:vo,:l,:h,:u)
            """),
            {
                "id":str(uuid.uuid4()),
                "b":req.bible,
                "bo":req.book,
                "c":req.chapter,
                "v":req.verse,
                "vo":req.voice,
                "l":req.language,
                "h":h,
                "u":url
            }
        )

    return {"audio_url":url,"cached":False}


# ======================
# GENERATE CHAPTER
# ======================

@app.post("/generate-chapter")
def generate_chapter(data):

    results=[]

    for verse in data["verses"]:

        req=VerseRequest(
            bible=data["bible"],
            book=data["book"],
            chapter=data["chapter"],
            verse=verse["verse"],
            voice=data["voice"],
            language=data["language"],
            text=verse["text"]
        )

        results.append(generate(req))

    return results


# ======================
# VOICE CLONE
# ======================

@app.post("/clone")
async def clone(file:UploadFile):

    path=f"{VOICE_DIR}/{file.filename}"

    with open(path,"wb") as f:
        f.write(await file.read())

    return {"status":"voice added"}


# ======================
# AI CHAT BIBLICO
# ======================

@app.post("/ai-chat")
def ai_chat(req:ChatRequest):

    response_text=f"Respuesta bíblica a: {req.message}"

    voice_path=f"{VOICE_DIR}/{req.voice}"

    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:

        tts.tts_to_file(
            text=response_text,
            speaker_wav=voice_path,
            language=req.language,
            file_path=tmp.name
        )

        object_name=f"chat_{uuid.uuid4()}.mp3"

        s3.upload_file(tmp.name,BUCKET,object_name)

    url=f"{PUBLIC_URL}/{object_name}"

    return {
        "text":response_text,
        "audio_url":url
    }


# ======================
# DYNAMIC READING
# ======================

@app.post("/dynamic-reading")
def dynamic_reading(data:DynamicReading):

    full_text=" ".join(data.verses)

    voice_path=f"{VOICE_DIR}/{data.voice}"

    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:

        tts.tts_to_file(
            text=full_text,
            speaker_wav=voice_path,
            language=data.language,
            file_path=tmp.name
        )

        object_name=f"reading_{uuid.uuid4()}.mp3"

        s3.upload_file(tmp.name,BUCKET,object_name)

    url=f"{PUBLIC_URL}/{object_name}"

    return {"audio_url":url}