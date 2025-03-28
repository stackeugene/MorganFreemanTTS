import os
import platform
import json
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, get_linear_schedule_with_warmup
from datasets import load_dataset
import logging
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TARGET_SAMPLE_RATE = 16000  # Required by SpeechT5
MEL_CHANNELS = 80
AUDIO_LENGTH_SECONDS = 5
TARGET_LENGTH = TARGET_SAMPLE_RATE * 10 # Changed to 10 seconds

# Helper Functions
def load_json_dataset(json_path: str) -> List[Dict]:
    """Loads dataset from a JSON file."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded dataset from {json_path}")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading JSON dataset: {e}")
        raise

def pad_or_truncate(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pads or truncates audio to a fixed length."""
    num_samples = waveform.shape[1]
    if num_samples < target_length:
        padding = target_length - num_samples
        waveform = F.pad(waveform, (0, padding))
    elif num_samples > target_length:
        waveform = waveform[:, :target_length]
    return waveform

def waveform_to_mel(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int = MEL_CHANNELS,
    hop_length: int = 256,  # Default value, can be adjusted
    win_length: int = 1024, # Default value, can be adjusted
    n_fft: int = 2048 # added n_fft
) -> torch.Tensor:
    """Converts waveform to mel spectrogram."""
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft
    )
    mel_spectrogram = mel_spectrogram_transform(waveform)
    return mel_spectrogram

def prepare_dataset(
    dataset: List[Dict],
    processor: SpeechT5Processor,
    target_length: int = TARGET_LENGTH
) -> List[Dict]:
    """Prepares dataset for training, handling audio loading, resampling, and mel spectrogram conversion."""
    processed_data = []
    for sample in dataset:
        audio_path = sample["audio_filepath"]
        text = sample["text"]
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
                )
                waveform = resampler(waveform)
            waveform = pad_or_truncate(waveform, target_length)
            mel_spectrogram = waveform_to_mel(waveform, TARGET_SAMPLE_RATE).squeeze(0)

            # Validate mel spectrogram shape
            if mel_spectrogram.shape[1] != MEL_CHANNELS:
                if mel_spectrogram.shape[1] > MEL_CHANNELS:
                    mel_spectrogram = mel_spectrogram[:, :MEL_CHANNELS]
                else:
                    pad_amount = MEL_CHANNELS - mel_spectrogram.shape[1]
                    mel_spectrogram = F.pad(mel_spectrogram, (0, pad_amount))

            inputs = processor(text=text, return_tensors="pt")
            input_ids = inputs["input_ids"].squeeze(0)
            attention_mask = inputs["attention_mask"].squeeze(0)

            processed_data.append({
                "mel_spectrogram": mel_spectrogram,
                "text": text,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            continue
    return processed_data

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collates data for a batch, padding sequences as necessary."""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    mel_spectrograms = [item["mel_spectrogram"] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    mel_spectrograms = torch.nn.utils.rnn.pad_sequence(mel_spectrograms, batch_first=True)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mel_spectrograms": mel_spectrograms,
    }

def fine_tune_model(
    train_dataset: List[Dict],
    validation_dataset: List[Dict],  # Added validation dataset
    num_epochs: int = 10,  # Increased number of epochs
    batch_size: int = 4,
    learning_rate: float = 2e-6,
    eval_steps: int = 50,
    save_steps: int = 200,
    output_dir: str = "fine_tuned_speecht5",
    gradient_accumulation_steps: int = 1,  # Add gradient accumulation
    max_grad_norm: float = 1.0, # Add max gradient norm
):
    """Fine-tunes the SpeechT5 model on the given dataset, with validation."""
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    validation_dataloader = DataLoader(  # Create validation dataloader
        validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps # adjust total_steps for gradient accumulation
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    best_eval_loss = float('inf') # Track the best validation loss
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        optimizer.zero_grad()  # Reset gradients at the beginning of each epoch
        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mel_spectrograms = batch["mel_spectrograms"].to(device)

            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=mel_spectrograms,
                )
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps # scale loss for accumulation
                loss.backward()

                if (i + 1) % gradient_accumulation_steps == 0: # update every gradient_accumulation_steps
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # Clip gradients
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"Train - Batch {i + 1}/{len(train_dataloader)} - Loss: {loss.item():.4f}, Learning Rate: {current_lr:.8f}"
                    )
            except Exception as e:
                logger.error(f"Error processing training batch {i + 1}: {e}")
                continue

        # Evaluate on the validation set periodically
        if (i + 1) % eval_steps == 0:
            eval_loss = evaluate_model(model, validation_dataloader, device)
            logger.info(f"Epoch {epoch+1} - Validation Loss: {eval_loss:.4f}")

            # Save the model if it has the best validation loss so far
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                logger.info(f"Best model saved at step {i+1} of epoch {epoch+1}")

        # Evaluate at the end of each epoch
        eval_loss = evaluate_model(model, validation_dataloader, device)
        logger.info(f"Epoch {epoch+1} - End of Epoch Validation Loss: {eval_loss:.4f}")
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            logger.info(f"Best model saved at the end of epoch {epoch+1}")

    # Save the final model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    logger.info("Fine-tuning complete. Model saved.")
    return processor, model  # Return the processor and model

def evaluate_model(model: SpeechT5ForTextToSpeech, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluates the model on the given dataloader and returns the average loss."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mel_spectrograms = batch["mel_spectrograms"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=mel_spectrograms,
            )
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)

def generate_speech(
    model: SpeechT5ForTextToSpeech,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    speaker_embeddings: torch.Tensor,
    output_path: str,
) -> None:
    """Generates speech from text and saves it as a WAV file."""
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        speech = model.generate_speech(
            input_ids, attention_mask=attention_mask, speaker_embeddings=speaker_embeddings
        )

    speech = torch.reshape(speech, (1, -1))
    torchaudio.save(output_path, speech.cpu(), TARGET_SAMPLE_RATE)
    absolute_path = os.path.abspath(output_path)
    logger.info(f"Generated speech saved to {absolute_path}")

if __name__ == "__main__":
    # Define dataset paths
    train_json_path = "SoundFiles/metadata_train.json"
    validation_json_path = "SoundFiles/metadata_eval.json"  # Changed to metadata_eval.json

    # Check if the validation file exists
    if not os.path.exists(validation_json_path):
        logger.error(f"Validation file not found: {validation_json_path}")
        raise FileNotFoundError(f"Validation file not found: {validation_json_path}")

    # Load datasets
    train_dataset = load_json_dataset(train_json_path)
    validation_dataset = load_json_dataset(validation_json_path)

    # Initialize processor
    model_name = "microsoft/speecht5_tts"
    processor = SpeechT5Processor.from_pretrained(model_name)

    # Prepare datasets
    train_processed_dataset = prepare_dataset(train_dataset, processor)
    validation_processed_dataset = prepare_dataset(validation_dataset, processor)

    # Fine-tune model
    processor, model = fine_tune_model(
        train_processed_dataset,
        validation_processed_dataset,
        num_epochs=1,  # Increased number of epochs
        batch_size=8,
        learning_rate=2e-6,
        eval_steps=50,
        save_steps=200,
        output_dir="fine_tuned_speecht5",
    )

    # Load speaker embeddings
    speaker_embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(speaker_embeddings_dataset[0]["xvector"]).unsqueeze(0)

    # Generate sample speech
    test_text = "Hello, I am Morgan Freeman."
    inputs = processor(text=test_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    generate_speech(
        model, input_ids, attention_mask, speaker_embeddings, "morgan_freeman_speech.wav"
    )

    # Play the generated speech (OS compatibility)
    if platform.system() == "Darwin":  # macOS
        os.system("open morgan_freeman_speech.wav")
    elif platform.system() == "Windows":  # Windows
        os.system("start morgan_freeman_speech.wav")
    else:
        logger.warning("Unsupported operating system for playing audio.")
