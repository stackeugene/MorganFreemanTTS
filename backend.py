import os
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from morgan13 import generate_speech, SpeechT5Processor, SpeechT5ForTextToSpeech
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = "./speecht5_finetuned"
embedding_file = "avg_speaker_embedding.pt"
processor = SpeechT5Processor.from_pretrained(output_dir)
model = SpeechT5ForTextToSpeech.from_pretrained(output_dir).to(device)
avg_speaker_embedding = torch.load(embedding_file)

# Define request model
class TextInput(BaseModel):
    text: str

# API endpoint to generate speech
@app.post("/generate_speech/")
async def generate_speech_endpoint(input: TextInput):
    # Process text input
    inputs = processor(text=input.text, return_tensors="pt")
    output_path = "morgan_freeman_output.wav"
    
    # Generate speech
    generate_speech(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        output_path,
        avg_speaker_embedding
    )
    
    # Return audio file
    return FileResponse(output_path, media_type="audio/wav", filename="morgan_freeman_output.wav")

# Run server: `uvicorn backend:app --reload`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)