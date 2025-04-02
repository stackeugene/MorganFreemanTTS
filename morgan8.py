import os
import platform
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, get_linear_schedule_with_warmup, AutoModel
import logging
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TARGET_SAMPLE_RATE = 16000
MEL_CHANNELS = 80
TARGET_LENGTH = TARGET_SAMPLE_RATE * 12  # 12 seconds of audio
HOP_LENGTH = 256  # Must match SpeechT5's decoder

def load_json_dataset(json_path: str) -> List[Dict]:
    """Load dataset from JSON file with error handling."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded dataset from {json_path}")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading JSON dataset: {e}")
        raise

def pad_or_truncate(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Ensure audio length is divisible by HOP_LENGTH (256 for SpeechT5)."""
    compatible_length = (target_length // HOP_LENGTH) * HOP_LENGTH
    current_length = waveform.shape[-1]
    
    if current_length < compatible_length:
        return F.pad(waveform, (0, compatible_length - current_length))
    elif current_length > compatible_length:
        return waveform[..., :compatible_length]
    return waveform

def waveform_to_mel(
    waveform: torch.Tensor,
    sample_rate: int = TARGET_SAMPLE_RATE,
    n_mels: int = MEL_CHANNELS,
    hop_length: int = HOP_LENGTH,
    win_length: int = 1024,
    n_fft: int = 2048,
) -> torch.Tensor:
    """Convert waveform to mel spectrogram with SpeechT5-compatible specs."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.shape[0] != 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft
    )
    mel = mel_transform(waveform)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel.squeeze(0)  # [80, time]

def prepare_dataset(
    dataset: List[Dict],
    processor: SpeechT5Processor,
    embedding_model=None,
    device="cpu"
) -> List[Dict]:
    """Prepare dataset with proper error handling and logging."""
    processed_data = []
    
    for sample in dataset:
        audio_path = sample["audio_filepath"]
        try:
            # 1. Load audio file
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.nelement() == 0:
                raise ValueError("Empty audio file")
            
            # 2. Convert to mono and resample if needed
            waveform = waveform.mean(dim=0, keepdim=True)  # [1, samples]
            if sample_rate != TARGET_SAMPLE_RATE:
                waveform = torchaudio.functional.resample(
                    waveform, 
                    orig_freq=sample_rate, 
                    new_freq=TARGET_SAMPLE_RATE
                )
            
            # 3. Pad/truncate to compatible length
            waveform = pad_or_truncate(waveform, TARGET_LENGTH)
            if waveform.shape[-1] % 256 != 0:
                raise ValueError(f"Invalid waveform length: {waveform.shape[-1]}")
            
            # 4. Generate mel spectrogram
            mel = waveform_to_mel(waveform)
            if mel.shape[0] != 80 or mel.shape[1] % 256 != 0:
                raise ValueError(f"Invalid mel shape: {mel.shape}")
            
            # 5. Process text
            inputs = processor(text=sample["text"], return_tensors="pt")
            
            # 6. Get speaker embedding if available
            speaker_embedding = None
            if embedding_model is not None:
                with torch.no_grad():
                    emb_input = waveform.squeeze(0).unsqueeze(0).to(device)
                    outputs = embedding_model(emb_input)
                    speaker_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            
            processed_data.append({
                "mel_spectrogram": mel.T,  # [time, 80]
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "speaker_embedding": speaker_embedding
            })
            
        except Exception as e:
            logger.error(f"Skipping {audio_path}: {str(e)}", exc_info=True)
            continue
            
    if not processed_data:
        raise RuntimeError("No valid audio files processed - check your dataset paths and file formats")
    
    # Debug first sample
    sample = processed_data[0]
    logger.info(f"First sample - Mel shape: {sample['mel_spectrogram'].shape}")
    logger.info(f"First sample - Text length: {sample['input_ids'].shape[0]}")
    
    return processed_data

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Batch preparation with proper padding."""
    mels = torch.nn.utils.rnn.pad_sequence(
        [x["mel_spectrogram"] for x in batch],
        batch_first=True
    ).transpose(1, 2)  # [batch, 80, time]
    
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in batch], batch_first=True),
        "mel_spectrograms": mels,
        "speaker_embeddings": torch.stack([x["speaker_embedding"] for x in batch])
            if batch[0]["speaker_embedding"] is not None else None
    }

def fine_tune_model(
    train_dataset: List[Dict],
    val_dataset: List[Dict],
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    output_dir: str = "./speecht5_finetuned"
):
    """Complete training loop with fixes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    embedding_model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    
    # Prepare data
    train_data = prepare_dataset(train_dataset, processor, embedding_model, device)
    val_data = prepare_dataset(val_dataset, processor, embedding_model, device)
    
    # Debug output
    sample = train_data[0]
    logger.info(f"Sample mel shape: {sample['mel_spectrogram'].shape} (time, 80)")
    logger.info(f"Sample text length: {sample['input_ids'].shape[0]}")
    
    # Dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * num_epochs
    )
    
    # Training
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["mel_spectrograms"].to(device),
                "speaker_embeddings": batch["speaker_embeddings"].to(device)
                    if batch["speaker_embeddings"] is not None else None
            }
            
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        # Validation
        val_loss = evaluate_model(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["mel_spectrograms"].to(device)
            }
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)

def generate_speech(
    model: SpeechT5ForTextToSpeech,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    output_path: str,
    speaker_embeddings: Optional[torch.Tensor] = None,
) -> None:
    model.eval()
    with torch.no_grad():
        speech = model.generate_speech(
            input_ids,
            attention_mask=attention_mask,
            speaker_embeddings=speaker_embeddings
        )
    torchaudio.save(output_path, speech.unsqueeze(0).cpu(), TARGET_SAMPLE_RATE)
    logger.info(f"Generated speech saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    # Load datasets
    train_data = load_json_dataset("SoundFiles/metadata_train.json")
    val_data = load_json_dataset("SoundFiles/metadata_eval.json")
    
    # Initialize processor and models
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    
    # Prepare datasets
    train_processed = prepare_dataset(train_data, processor, embedding_model, device)
    val_processed = prepare_dataset(val_data, processor, embedding_model, device)
    
    # Debug output
    sample = train_processed[0]
    print(f"\nDebug Shape Check:")
    print(f"Mel shape: {sample['mel_spectrogram'].shape} (time, 80)")
    print(f"Text length: {sample['input_ids'].shape[0]}")
    if sample["speaker_embedding"] is not None:
        print(f"Speaker embedding: {sample['speaker_embedding'].shape}")
    
    # Train
    fine_tune_model(
        train_processed,
        val_processed,
        num_epochs=5,
        batch_size=8,
        learning_rate=3e-5
    )
    
    # Generate sample
    test_text = "Hello, I am Morgan Freeman."
    inputs = processor(text=test_text, return_tensors="pt")
    generate_speech(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        "morgan_freeman.wav"
    )
    
    # Play audio
    if platform.system() == "Darwin":
        os.system("open morgan_freeman.wav")
    elif platform.system() == "Windows":
        os.system("start morgan_freeman.wav")
