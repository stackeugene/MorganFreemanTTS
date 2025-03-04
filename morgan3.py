import os
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from datasets import load_dataset

# Load dataset from JSON file
def load_json_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Pad or truncate audio to a fixed length
def pad_or_truncate(waveform, target_length):
    num_samples = waveform.shape[1]
    
    if num_samples < target_length:
        # Pad with zeros if too short
        padding = target_length - num_samples
        waveform = F.pad(waveform, (0, padding))  # Pad at the end
    elif num_samples > target_length:
        # Truncate if too long
        waveform = waveform[:, :target_length]
    return waveform

# Convert waveform to mel spectrogram
def waveform_to_mel(waveform, sample_rate, n_mels=80):
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels
    )
    return mel_spectrogram_transform(waveform)

# Prepare dataset for training
def prepare_dataset(dataset, target_length=16000 * 5):  # 5 seconds of audio at 16kHz
    processed_data = []
    target_sample_rate = 16000  # Required by SpeechT5

    for sample in dataset:
        print(sample)
        audio_path = sample["audio_filepath"]
        print(f"Loading audio file: {audio_path}")  # Print the file path
        text = sample["text"]

        # Load audio waveform
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Apply padding/truncation
        waveform = pad_or_truncate(waveform, target_length)

        # Convert to mel spectrogram
        mel_spectrogram = waveform_to_mel(waveform, target_sample_rate).squeeze(0)  # Remove batch dimension if present

        # Ensure the mel spectrogram has the correct shape [sequence_length, num_mel_bins]
        if mel_spectrogram.shape[1] > 400:
            mel_spectrogram = mel_spectrogram[:, :400]  # Truncate to 400 time steps
        elif mel_spectrogram.shape[1] < 400:
            pad_amount = 400 - mel_spectrogram.shape[1]
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, 0, 0, pad_amount))  # Pad time dimension

        # Transpose to [sequence_length, num_mel_bins]
        mel_spectrogram = mel_spectrogram.permute(1, 0)

        processed_data.append({
            "mel_spectrogram": mel_spectrogram,
            "text": text
        })

    return processed_data

# Fine-tuning function for SpeechT5
def fine_tune_model(dataset, num_epochs=5, batch_size=2, learning_rate=5e-5):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Convert dataset into tensors and use DataLoader
    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        mel_spectrograms = [item["mel_spectrogram"] for item in batch]

        # Ensure all mel spectrograms have the same shape [sequence_length, num_mel_bins]
        mel_spectrograms = [F.pad(mel, (0, 0, 0, 400 - mel.shape[0])) if mel.shape[0] < 400 else mel[:400, :] for mel in mel_spectrograms]
        mel_spectrograms = torch.stack(mel_spectrograms)  # Stack into a batch

        return {"mel_spectrograms": mel_spectrograms, "texts": texts}

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        for i, sample in enumerate(dataloader):
            mel_spectrograms = sample["mel_spectrograms"]  # Already extracted from dataset
            texts = sample["texts"]

            try:
                # Process text inputs
                text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                text_inputs["attention_mask"] = text_inputs["attention_mask"].clone().detach().float()

                # Debugging: Print shapes before passing to the model
                print(f"Text input IDs shape: {text_inputs['input_ids'].shape}")
                print(f"Attention mask shape: {text_inputs['attention_mask'].shape}")
                print(f"Mel spectrograms shape: {mel_spectrograms.shape}")

                # Ensure spectrograms are correctly shaped for the model
                mel_spectrograms = mel_spectrograms.permute(0, 2, 1)  # Shape [batch_size, num_mel_bins, sequence_length]
                print(f"Adjusted Mel spectrograms shape: {mel_spectrograms.shape}")

                # Forward pass
                outputs = model(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                    labels=mel_spectrograms
                )

                # Compute loss
                loss = outputs.loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Batch {i+1}/{len(dataloader)} - Loss: {loss.item()}")

            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
                continue

        print(f"Epoch {epoch+1} complete.\n")

    # Save fine-tuned model
    model.save_pretrained("fine_tuned_speecht5")
    processor.save_pretrained("fine_tuned_speecht5")
    print("Fine-tuning complete. Model saved.")


def generate_speech(model, input_ids, attention_mask, speaker_embeddings, output_path):
    # Generate speech
    with torch.no_grad():
        speech = model.generate_speech(input_ids=input_ids, attention_mask=attention_mask, speaker_embeddings=speaker_embeddings)
    
    # Save the generated speech to a file
    with open(output_path, "wb") as f:
        f.write(speech)

# Define dataset path
dataset_path = "morgan_freeman_tts_dataset"

if __name__ == "__main__":
    # Load dataset
    train_json_path = r"SoundFiles/metadata_train.json"
    dataset = load_json_dataset(train_json_path)

    # Prepare dataset
    processed_dataset = prepare_dataset(dataset)

    # Fine-tune model
    fine_tune_model(processed_dataset)

    # Initialize tokenizer and model
    model_name = "microsoft/speecht5_tts"  # Valid model name
    processor = SpeechT5Processor.from_pretrained(model_name)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_name)

    # Load speaker embeddings
    speaker_embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(speaker_embeddings_dataset[0]["xvector"]).unsqueeze(0)

    # Generate sample speech, play it afterwards
    inputs = processor(text="Hello, I am Morgan Freeman.", return_tensors="pt")
    input_ids = inputs["input_ids"]  # Do not convert to float tensor
    attention_mask = inputs["attention_mask"].float()  # Convert to float tensor
    generate_speech(model, input_ids, attention_mask, speaker_embeddings, "morgan_freeman_speech.wav")
    os.system("open morgan_freeman_speech.wav")
