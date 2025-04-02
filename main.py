from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from enum import Enum
from transformers import pipeline, MarianMTModel, MarianTokenizer
import shutil
import os
import uuid
from googletrans import Translator

app = FastAPI()

# 🌍 Language Enum for dropdown in Swagger
class LanguageEnum(str, Enum):
    ta = "ta"  # Tamil
    fr = "fr"  # French
    es = "es"  # Spanish
    de = "de"  # German
    it = "it"  # Italian
    hi = "hi"  # Hindi
    ru = "ru"  # Russian
    zh = "zh"  # Chinese
    ar = "ar"  # Arabic

# 🌐 Map target language to translation model
model_map = {
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "de": "Helsinki-NLP/opus-mt-en-de",
    "it": "Helsinki-NLP/opus-mt-en-it",
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "ru": "Helsinki-NLP/opus-mt-en-ru",
    "zh": "Helsinki-NLP/opus-mt-en-zh",
    "ar": "Helsinki-NLP/opus-mt-en-ar",
    "ta": "gsarti/opus-mt-en-ta"
}

def translate_text(text, target_lang):
    if target_lang == "ta":
        try:
            translator = Translator()
            result = translator.translate(text, dest="ta")
            return result.text
        except Exception as e:
            return f"Google Translate failed: {str(e)}"

    if target_lang not in model_map:
        return f"No model for language: {target_lang}"

    model_name = model_map[target_lang]
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    encoded = tokenizer([text], return_tensors="pt", padding=True)
    translated = model.generate(**encoded)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

# 🧠 Generate a random English sentence
def generate_random_sentence(prompt="Daily conversation", max_length=30):
    generator = pipeline("text-generation", model="distilgpt2")
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    return result[0]["generated_text"].strip()

# 🎤 Transcription endpoint
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    with open(temp_filename, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    try:
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-medium")
        result = asr(temp_filename)
        return JSONResponse(content={"transcribed_text": result["text"]})
    finally:
        os.remove(temp_filename)

# 🌍 Translation endpoint
@app.post("/translate")
async def translate(text: str = Form(...), target_lang: LanguageEnum = Form(...)):
    translated = translate_text(text, target_lang.value)
    return JSONResponse(content={"translated_text": translated})

# 🔁 Combined endpoint (speech-to-translation)
@app.post("/process")
async def process(audio: UploadFile = File(...), target_lang: LanguageEnum = Form(...)):
    temp_filename = f"temp_{uuid.uuid4().hex}.wav"
    with open(temp_filename, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    try:
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-medium")
        result = asr(temp_filename)
        transcribed_text = result["text"]
        translated_text = translate_text(transcribed_text, target_lang.value)
        return JSONResponse(content={
            "transcribed_text": transcribed_text,
            "translated_text": translated_text
        })
    finally:
        os.remove(temp_filename)

# ✨ Generate + Translate endpoint
@app.get("/generate")
def generate(prompt: str = "Daily conversation", target_lang: LanguageEnum = LanguageEnum.it):
    english = generate_random_sentence(prompt)
    translated = translate_text(english, target_lang.value)
    return {
        "prompt": prompt,
        "english": english,
        "translated": translated
    }
