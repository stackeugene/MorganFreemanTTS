import os
import json
import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech

# Load dataset from JSON file
def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Prepare dataset for training
def prepare_dataset(dataset):
    processed_data = []
    for sample in dataset:
        print(sample)
        audio_path = sample["audio_filepath"]
        print(f"Loading audio file: {audio_path}")  # Print the file path
        text = sample["text"]
        
        # Load audio waveform
        waveform, sample_rate = torchaudio.load(audio_path)
        
        processed_data.append({
            "waveform": waveform,
            "sample_rate": sample_rate,
            "text": text
        })
    return processed_data

# Train a SpeechT5 model (basic fine-tuning)
def train_model(dataset):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    target_sample_rate = 16000  # Required by SpeechT5

    for i, sample in enumerate(dataset):
        print(f"Processing sample {i+1}: {sample}")  # Debugging line

        # Extract waveform and sample rate
        waveform = sample["waveform"]
        sample_rate = sample["sample_rate"]

        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Process text (ONLY text is needed for training)
        text_inputs = processor(text=sample["text"].strip(), return_tensors="pt")
        if "input_ids" not in text_inputs or text_inputs["input_ids"] is None:
            raise ValueError(f"Processor failed to generate input_ids for text: {sample['text']}")

        print("Text Processor output:", text_inputs)

        # Pass only text-based inputs to the model
        inputs = {key: value for key, value in text_inputs.items() if value is not None}
        outputs = model(**inputs)  # No audio inputs are passed

        print(f"Processed sample {i+1}/{len(dataset)}")

    print("Training complete.")

# Generate speech from text
def generate_speech(text, output_path="output.wav"):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    inputs = processor(text, return_tensors="pt")
    speech = model.generate(**inputs)

    torchaudio.save(output_path, speech, 22050)
    print(f"Generated speech saved to {output_path}")

if __name__ == "__main__":
    dataset_path = "morgan_freeman_tts_dataset"
    
    # Load dataset (TO ALL OF GROUP 2 - NEED TO REPLACE THIS WITH YOUR OWN FILE PATH TO THIS FILE)
    train_json_path = r"SoundFiles\metadata_train.json"
    dataset = load_dataset(train_json_path)

    # Prepare dataset
    processed_dataset = prepare_dataset(dataset)

    # Train model
    train_model(processed_dataset)

    # Generate sample speech
    generate_speech("Hello, I am Morgan Freeman.", "morgan_freeman_speech.wav")
