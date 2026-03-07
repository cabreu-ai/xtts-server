from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from TTS.api import TTS
import uuid
import os

app = FastAPI()

templates = Jinja2Templates(directory="templates")

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

tts = TTS(MODEL)

VOICE_DIR = "/app/voices"
OUTPUT_DIR = "/app/outputs"

os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/clone")
async def clone_voice(file: UploadFile = File(...)):

    voice_id = str(uuid.uuid4())
    path = f"{VOICE_DIR}/{voice_id}.wav"

    with open(path, "wb") as f:
        f.write(await file.read())

    return {"voice_id": voice_id}


@app.post("/tts")
async def generate(text: str = Form(...), voice_id: str = Form(...)):

    voice_path = f"{VOICE_DIR}/{voice_id}.wav"
    output = f"{OUTPUT_DIR}/{uuid.uuid4()}.wav"

    tts.tts_to_file(
        text=text,
        speaker_wav=voice_path,
        language="es",
        file_path=output
    )

    return FileResponse(output)