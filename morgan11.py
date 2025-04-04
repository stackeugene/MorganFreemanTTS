import os
import platform
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, get_linear_schedule_with_warmup, AutoModel, SpeechT5HifiGan, EarlyStoppingCallback
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
    num_epochs: int,  # These variables must be instantiated but not recommended to change
    batch_size: int, # these will just get overwritten anyway, change elsewhere.
    learning_rate: float,
    output_dir: str = "./speecht5_finetuned",
    resume_training: bool = False
) -> SpeechT5ForTextToSpeech:
    """Complete training loop with resume capability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize components
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    
    # Load model with resume support
    if resume_training and os.path.exists(output_dir):
        logger.info(f"Resuming training from {output_dir}")
        model = SpeechT5ForTextToSpeech.from_pretrained(output_dir).to(device)
        
        # Load optimizer state if available
        optimizer_state_path = os.path.join(output_dir, "optimizer.pt")
        if os.path.exists(optimizer_state_path):
            optimizer_state = torch.load(optimizer_state_path)
        else:
            optimizer_state = None
    else:
        logger.info("Starting new training session")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
        optimizer_state = None

    # Prepare datasets
    if all(key in train_dataset[0] for key in ["mel_spectrogram", "input_ids", "attention_mask"]):
        train_data = train_dataset
        val_data = val_dataset
        logger.info("Using preprocessed dataset directly")
    else:
        embedding_model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        train_data = prepare_dataset(train_dataset, processor, embedding_model, device)
        val_data = prepare_dataset(val_dataset, processor, embedding_model, device)
    
    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn)
    
    # Optimizer with resume support
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    
    # Training state initialization
    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0
    patience = 5
    threshold = 0.01
    
    # Load training state if resuming
    if resume_training:
        training_state_path = os.path.join(output_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            start_epoch = training_state["epoch"]
            best_loss = training_state["best_loss"]
            patience_counter = training_state["patience_counter"]
            logger.info(f"Resuming from epoch {start_epoch} with best loss {best_loss:.4f}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * num_epochs
    )
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["mel_spectrograms"].permute(0, 2, 1).to(device),
                "speaker_embeddings": None
            }
            
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        val_loss = evaluate_model(model, val_loader, device)
        logger.info(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping and checkpointing
        if val_loss < best_loss - threshold:
            best_loss = val_loss
            patience_counter = 0
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save({
                "epoch": epoch + 1,
                "best_loss": best_loss,
                "patience_counter": patience_counter
            }, os.path.join(output_dir, "training_state.pt"))
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["mel_spectrograms"].permute(0, 2, 1).to(device),
                "speaker_embeddings": None
            }
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
    
    # Load vocoder
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(model.device)
    
    # Configure generation parameters properly
    if hasattr(model, 'generation_config'):
        # Correct way to update generation config
        model.generation_config.update(
            do_sample=True,
            temperature=0.7,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            return_output_lengths=True
        )
    else:
        # Fallback if generation_config doesn't exist
        model.generation_config = {
            'do_sample': True,
            'temperature': 0.7,
            'repetition_penalty': 2.0,
            'no_repeat_ngram_size': 3,
            'return_output_lengths': True
        }
    
    with torch.no_grad():
        # Warmup generation
        _ = model.generate_speech(
            input_ids[:,:10],
            attention_mask=attention_mask[:,:10] if attention_mask is not None else None,
            speaker_embeddings=speaker_embedding.unsqueeze(0).to(model.device)
        )
        
        # Generate mel spectrogram
        generation_output = model.generate_speech(
            input_ids,
            attention_mask=attention_mask,
            speaker_embeddings=speaker_embedding.unsqueeze(0).to(model.device)
        )
            
        # Handle tuple output (spectrogram, lengths) if returned
        if isinstance(generation_output, tuple):
            mel_spectrogram, output_lengths = generation_output
            logger.info(f"Generated {output_lengths.item()} frames")
        else:
            mel_spectrogram = generation_output
            logger.info(f"Mel spectrogram shape: {mel_spectrogram.shape}")
            
            # Convert to waveform
        speech = vocoder(mel_spectrogram)
            
        # Ensure proper shape [channels, samples]
        if speech.dim() == 1:
            speech = speech.unsqueeze(0)  # [samples] -> [1, samples]
        elif speech.dim() == 3:
            speech = speech.squeeze(0)  # [1, channels, samples] -> [channels, samples]
            
        # Verify and normalize audio
        if speech.abs().max() < 1e-6:
            raise ValueError("Generated audio is silent")
        speech = speech / (speech.abs().max() + 1e-7)
            
        # Trim trailing silence
        speech = trim_silence(speech, sample_rate=TARGET_SAMPLE_RATE)
            
        # Calculate final duration
        duration = speech.shape[-1] / TARGET_SAMPLE_RATE
        logger.info(f"Final speech duration: {duration:.2f} seconds")
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            
        # Save with fallback options
        try:
            torchaudio.save(output_path, speech.cpu(), TARGET_SAMPLE_RATE)
        except Exception as e:
            logger.warning(f"Primary save failed, trying backup method: {str(e)}")
            import soundfile as sf
            sf.write(
                output_path,
                speech.squeeze().cpu().numpy(),
                TARGET_SAMPLE_RATE
            )
                
        except Exception as e:
            logger.error(f"Audio generation failed: {str(e)}")
            raise
        
    logger.info(f"Successfully saved to {os.path.abspath(output_path)}")

def trim_silence(waveform: torch.Tensor, sample_rate: int, threshold_db: float = -40) -> torch.Tensor:
    """Trim trailing silence from waveform using energy threshold."""
    import numpy as np
    
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Convert to numpy for processing
    audio_np = waveform.squeeze().cpu().numpy()
    
    # Calculate energy
    energy = np.abs(audio_np)
    threshold = 10 ** (threshold_db / 20) * energy.max()
    
    # Find last point above threshold
    if np.any(energy > threshold):
        last_non_silent = np.where(energy > threshold)[0][-1]
    else:
        last_non_silent = len(audio_np)  # Keep entire waveform if no silence found
    
    # Add 100ms buffer
    buffer_samples = int(0.1 * sample_rate)
    end_point = min(last_non_silent + buffer_samples, len(audio_np))
    return waveform[..., :end_point]

if __name__ == "__main__":
    test_text = input("Please input what you would like Morgan Freeman to say: ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "./speecht5_finetuned"
    embedding_file = "avg_speaker_embedding.pt"
    
    # Training options
    print(f"\n{'='*50}")
    print(f"Found model at: {os.path.abspath(output_dir)}" if os.path.exists(output_dir) else "No existing model found")
    print("="*50)
    
    while True:
        choice = input("\nOptions:\n"
                      "1. Train new model\n"
                      "2. Resume training\n"
                      "3. Skip training and generate only\n"
                      "Enter choice (1/2/3): ").strip()
        
        if choice in ('1', '2', '3'):
            break
        print("Invalid input! Please enter 1, 2, or 3")

    if choice != '3':
        # Load and prepare datasets
        train_data = load_json_dataset("SoundFiles/metadata_train.json")
        val_data = load_json_dataset("SoundFiles/metadata_eval.json")
        
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        embedding_model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        
        train_processed = prepare_dataset(train_data, processor, embedding_model, device)
        val_processed = prepare_dataset(val_data, processor, embedding_model, device)
        
        # Train with selected option
        model = fine_tune_model(
            train_processed,
            val_processed,
            num_epochs=30, # this is what should be changed if epochs need to change
            batch_size=2,
            learning_rate=5e-6,
            output_dir=output_dir,
            resume_training=(choice == '2')
        )
        
        # Compute and save speaker embedding
        avg_speaker_embedding = compute_average_speaker_embedding(train_processed)
        torch.save(avg_speaker_embedding, embedding_file)
    
    # Generation block
    if os.path.exists(output_dir):
        print("\nGenerating sample audio...")
        processor = SpeechT5Processor.from_pretrained(output_dir)
        model = SpeechT5ForTextToSpeech.from_pretrained(output_dir).to(device)
        
        if os.path.exists(embedding_file):
            avg_speaker_embedding = torch.load(embedding_file)
        else:
            raise FileNotFoundError(f"Speaker embedding file {embedding_file} not found")
        
        inputs = processor(text=test_text, return_tensors="pt")
        model.generation_config.max_length = inputs["input_ids"].shape[-1] + 100
        
        generate_speech(
            model,
            inputs["input_ids"],
            inputs["attention_mask"],
            "morgan_freeman.wav",
            avg_speaker_embedding
        )
        
        # Play result
        print("\nGeneration complete! Playing audio...")
        if platform.system() == "Darwin":
            os.system("open morgan_freeman.wav")
        elif platform.system() == "Windows":
            os.system("start morgan_freeman.wav")
