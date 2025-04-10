import os
import platform
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, get_linear_schedule_with_warmup, AutoModel, SpeechT5HifiGan
import logging
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TARGET_SAMPLE_RATE = 16000
MEL_CHANNELS = 80
HOP_LENGTH = 256  # Must match SpeechT5's decoder
DESIRED_FRAMES = 768  # Multiple of 256, close to original 12 seconds
TARGET_LENGTH = (DESIRED_FRAMES - 1) * HOP_LENGTH + 2048  # 198,400 samples

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

def pad_or_truncate(waveform: torch.Tensor, target_length: int = TARGET_LENGTH) -> torch.Tensor:
    """Ensure audio length produces exactly DESIRED_FRAMES mel frames."""
    current_length = waveform.shape[-1]
    
    if current_length != target_length:
        if current_length < target_length:
            waveform = F.pad(waveform, (0, target_length - current_length))
        else:
            waveform = waveform[..., :target_length]
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
    logger.info(f"Raw mel shape: {mel.shape}")
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mel = mel.squeeze(0)  # [80, time]
    
    # Truncate to exactly DESIRED_FRAMES if necessary
    if mel.shape[1] > DESIRED_FRAMES:
        mel = mel[:, :DESIRED_FRAMES]
    elif mel.shape[1] < DESIRED_FRAMES:
        mel = F.pad(mel, (0, DESIRED_FRAMES - mel.shape[1]))
    
    return mel  # [80, 768]

def prepare_dataset(
    dataset: List[Dict],
    processor: SpeechT5Processor,
    embedding_model=None,
    device="cpu"
) -> List[Dict]:
    """Prepare dataset with proper error handling and logging."""
    processed_data = []
    
    for sample in dataset:
        try:
            logger.info(f"Sample keys: {sample.keys()}")  # Debug dataset structure
            audio_path = sample["audio_filepath"]  # Replace with correct key, e.g., "path"
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
            logger.info(f"Waveform length after pad/truncate: {waveform.shape[-1]}")
            
            # 4. Generate mel spectrogram
            mel = waveform_to_mel(waveform)
            logger.info(f"Mel shape before check: {mel.shape}")
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
            logger.error(f"Skipping sample with keys {sample.keys()}: {str(e)}", exc_info=True)
            continue
            
    if not processed_data:
        raise RuntimeError("No valid audio files processed - check your dataset paths and file formats")
    
    # Debug first sample
    sample = processed_data[0]
    logger.info(f"First sample - Mel shape: {sample['mel_spectrogram'].shape}")
    logger.info(f"First sample - Text length: {sample['input_ids'].shape[0]}")
    
    return processed_data

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Batch preparation with proper padding and shape correction."""
    mels = torch.nn.utils.rnn.pad_sequence(
        [x["mel_spectrogram"].T for x in batch],  # [768, 80] to [80, 768]
        batch_first=True
    )  # [batch, 80, time]
    
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in batch], batch_first=True),
        "mel_spectrograms": mels,
        # Omit speaker_embeddings from collation
    }

def fine_tune_model(
    train_dataset: List[Dict],
    val_dataset: List[Dict],
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    output_dir: str = "./speecht5_finetuned"
) -> SpeechT5ForTextToSpeech:  # Add return type
    """Complete training loop with fixes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    
    if all(key in train_dataset[0] for key in ["mel_spectrogram", "input_ids", "attention_mask"]):
        train_data = train_dataset
        val_data = val_dataset
        logger.info("Using preprocessed dataset directly")
    else:
        embedding_model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        train_data = prepare_dataset(train_dataset, processor, embedding_model, device)
        val_data = prepare_dataset(val_dataset, processor, embedding_model, device)
    
    sample = train_data[0]
    logger.info(f"Sample mel shape: {sample['mel_spectrogram'].shape} (time, 80)")
    logger.info(f"Sample text length: {sample['input_ids'].shape[0]}")
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * num_epochs
    )
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            logger.info(f"Batch mel shape: {batch['mel_spectrograms'].shape}")
            optimizer.zero_grad()
            
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["mel_spectrograms"].permute(0, 2, 1).to(device),  # [8, 768, 80]
                "speaker_embeddings": None
            }
            
            logger.info(f"Inputs to model: {[(k, v.shape) for k, v in inputs.items() if v is not None]}")
            try:
                outputs = model(**inputs)
                logger.info(f"Outputs loss: {outputs.loss.item()}")
            except Exception as e:
                logger.error(f"Model forward failed: {str(e)}", exc_info=True)
                raise
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        val_loss = evaluate_model(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
    
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["mel_spectrograms"].permute(0, 2, 1).to(device),  # [batch_size, 768, 80]
                "speaker_embeddings": None
            }
            logger.info(f"Eval inputs: {[(k, v.shape) for k, v in inputs.items() if v is not None]}")
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)

def compute_average_speaker_embedding(dataset: List[Dict]) -> torch.Tensor:
    """Compute an average speaker embedding from the dataset."""
    embeddings = [sample["speaker_embedding"] for sample in dataset if sample["speaker_embedding"] is not None]
    if not embeddings:
        raise ValueError("No speaker embeddings found in dataset")
    stacked_embeddings = torch.stack(embeddings)  # [N, 768]
    avg_embedding = torch.mean(stacked_embeddings, dim=0)  # [768]
    # Transform from 768 to 512 (SpeechT5 expects [512])
    linear_transform = torch.nn.Linear(768, 512)
    return linear_transform(avg_embedding).detach()  # [512]

def generate_speech(
    model: SpeechT5ForTextToSpeech,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    output_path: str,
    speaker_embedding: torch.Tensor,
) -> None:
    model.eval()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(model.device)
    with torch.no_grad():
        # Generate mel spectrogram
        mel_spectrogram = model.generate_speech(
            input_ids,
            attention_mask=attention_mask,
            speaker_embeddings=speaker_embedding.unsqueeze(0).to(model.device),  # [1, 512]
        )
        logger.info(f"Generated mel spectrogram shape: {mel_spectrogram.shape}")
        logger.info(f"Mel spectrogram content (first 10 samples): {mel_spectrogram[:10]}")
        
        # Convert to waveform using HiFi-GAN
        speech = vocoder(mel_spectrogram)
        logger.info(f"Generated waveform shape: {speech.shape}")
        logger.info(f"Waveform content (first 10 samples): {speech[:10]}")
        
        # Ensure speech is 2D [channels, samples]
        if speech.dim() == 1:  # [samples] -> [1, samples]
            speech = speech.unsqueeze(0)
        elif speech.dim() > 2:  # [1, 1, samples] -> [1, samples]
            speech = speech.squeeze(0) if speech.shape[0] == 1 else speech[0]
        if speech.shape[1] == 0:
            logger.warning("Generated speech has zero samples!")
        torchaudio.save(output_path, speech.cpu(), TARGET_SAMPLE_RATE)
    logger.info(f"Generated speech saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    # Load datasets
    train_data = load_json_dataset("SoundFiles/metadata_train.json")
    val_data = load_json_dataset("SoundFiles/metadata_eval.json")
    
    # Initialize processor
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
    
    # Train and get the model
    model = fine_tune_model(
        train_processed,
        val_processed,
        num_epochs=15,  # Increase to 15 epochs
        batch_size=8,
        learning_rate=3e-5
    )
    
    # Compute average speaker embedding
    avg_speaker_embedding = compute_average_speaker_embedding(train_processed)
    
    # Generate sample
    test_text = "Hello, I am Morgan Freeman."
    inputs = processor(text=test_text, return_tensors="pt")
    generate_speech(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        "morgan_freeman.wav",
        avg_speaker_embedding
    )
    
    # Play audio
    if platform.system() == "Darwin":
        os.system("open morgan_freeman.wav")
    elif platform.system() == "Windows":
        os.system("start morgan_freeman.wav")