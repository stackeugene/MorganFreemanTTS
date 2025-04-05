import os
import platform
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from speechbrain.pretrained import EncoderClassifier
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
    embedding_model=None,  # Will be initialized with SpeechBrain if None
    device="cpu"
) -> List[Dict]:
    """Prepare dataset with proper error handling and logging."""
    from speechbrain.pretrained import EncoderClassifier
    
    # Initialize SpeechBrain embedding model if not provided
    if embedding_model is None:
        embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb"
        ).to(device)
    
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
            
            # 6. Get speaker embedding using SpeechBrain
            speaker_embedding = None
            with torch.no_grad():
                emb_input = waveform.to(device)  # [1, samples]
                speaker_embedding = embedding_model.encode_batch(emb_input).squeeze(0).cpu()  # [512]
            
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
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in batch], batch_first=True
        ),
        "mel_spectrograms": torch.nn.utils.rnn.pad_sequence(
            [x["mel_spectrogram"].T for x in batch], batch_first=True
        ),  # [batch, 80, 768]
        "speaker_embeddings": torch.stack(
            [x["speaker_embedding"] for x in batch]
        )  # [batch, 512]
    }

def fine_tune_model(
    train_dataset: List[Dict],
    val_dataset: List[Dict],
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    output_dir: str = "./speecht5_finetuned"
) -> SpeechT5ForTextToSpeech:
    """Complete training loop with fixes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_loss = float('inf')
    patience_counter = 0
    patience = 5
    threshold = 0.01
    
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    
    # Use SpeechBrain embedding model
    embedding_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb"
    ).to(device)
    
    if all(key in train_dataset[0] for key in ["mel_spectrogram", "input_ids", "attention_mask", "speaker_embedding"]):
        train_data = train_dataset
        val_data = val_dataset
        logger.info("Using preprocessed dataset directly")
    else:
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
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            logger.info(f"Batch mel shape: {batch['mel_spectrograms'].shape}")
            optimizer.zero_grad()
            
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["mel_spectrograms"].permute(0, 2, 1).to(device),  # [batch, 768, 80]
                "speaker_embeddings": batch["speaker_embeddings"].to(device)  # [batch, 512]
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
        
        if val_loss < best_loss - threshold:
            best_loss = val_loss
            patience_counter = 0
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            logger.info(f"New best model saved (loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
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
                "speaker_embeddings": batch["speaker_embeddings"].to(device)
            }
            outputs = model(**inputs)
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)

def compute_average_speaker_embedding(dataset: List[Dict], device="cpu") -> torch.Tensor:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/spkrec-xvect-voxceleb"
    ).to(device)
    embeddings = []
    for sample in dataset:
        waveform, sr = torchaudio.load(sample["audio_filepath"])
        if sr != TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SAMPLE_RATE)
        waveform = waveform.mean(dim=0, keepdim=True).to(device)
        with torch.no_grad():
            emb = classifier.encode_batch(waveform).squeeze(0).cpu()
        embeddings.append(emb)
    if not embeddings:
        raise ValueError("No valid embeddings computed")
    return torch.mean(torch.stack(embeddings), dim=0)  # [512]

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "./speecht5_finetuned"
    embedding_file = "avg_speaker_embedding.pt"
    
    if os.path.exists(output_dir):
        print(f"\n{'='*50}")
        print(f"Found pretrained model at: {os.path.abspath(output_dir)}")
        print("="*50)
        
        while True:
            response = input("\nWould you like to:\n"
                           "1. Skip training and use existing model (fast)\n"
                           "2. Retrain new model (slow)\n"
                           "Enter choice (1/2): ").strip()
            if response in ('1', '2'):
                skip_training = (response == '1')
                break
            print("Invalid input! Please enter 1 or 2")
        
        if skip_training:
            logger.info("Loading pretrained model...")
            try:
                processor = SpeechT5Processor.from_pretrained(output_dir)
                model = SpeechT5ForTextToSpeech.from_pretrained(output_dir).to(device)
                if os.path.exists(embedding_file):
                    logger.info("Loading speaker embeddings...")
                    avg_speaker_embedding = torch.load(embedding_file)
                else:
                    logger.info("Computing speaker embeddings...")
                    train_data = load_json_dataset("SoundFiles/metadata_train.json")
                    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
                    embedding_model = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-xvect-voxceleb",
                        savedir="pretrained_models/spkrec-xvect-voxceleb"
                    ).to(device)
                    train_processed = prepare_dataset(train_data, processor, embedding_model, device)
                    avg_speaker_embedding = compute_average_speaker_embedding(train_processed, device)
                    torch.save(avg_speaker_embedding, embedding_file)
            except Exception as e:
                logger.error(f"Failed to load pretrained model: {str(e)}")
                print("\nFalling back to training new model...")
                skip_training = False
    
    if not skip_training:
        print("\nPreparing to train new model...")
        logger.info("Loading and processing datasets...")
        
        train_data = load_json_dataset("SoundFiles/metadata_train.json")
        val_data = load_json_dataset("SoundFiles/metadata_eval.json")
        
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        embedding_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb"
        ).to(device)
        
        train_processed = prepare_dataset(train_data, processor, embedding_model, device)
        val_processed = prepare_dataset(val_data, processor, embedding_model, device)
        
        sample = train_processed[0]
        print("\nDataset Verification:")
        print(f"- Mel spectrogram shape: {sample['mel_spectrogram'].shape} (time, 80)")
        print(f"- Text sequence length: {sample['input_ids'].shape[0]}")
        if sample["speaker_embedding"] is not None:
            print(f"- Speaker embedding size: {sample['speaker_embedding'].shape}")
        
        print("\nStarting training...")
        model = fine_tune_model(
            train_processed,
            val_processed,
            num_epochs=30,
            batch_size=2,
            learning_rate=5e-6,
            output_dir=output_dir
        )
        
        avg_speaker_embedding = compute_average_speaker_embedding(train_processed, device)
        torch.save(avg_speaker_embedding, embedding_file)
    
    print("\nGenerating sample audio...")
    test_text = "Hello, I am Morgan Freeman."
    inputs = processor(text=test_text, return_tensors="pt")
    model.generation_config.max_length = inputs["input_ids"].shape[-1] + 100
    
    generate_speech(
        model,
        inputs["input_ids"],
        inputs["attention_mask"],
        "morgan_freeman.wav",
        avg_speaker_embedding
    )
    
    print("\nGeneration complete! Playing audio...")
    if platform.system() == "Darwin":
        os.system("open morgan_freeman.wav")
    elif platform.system() == "Windows":
        os.system("start morgan_freeman.wav")