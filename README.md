# Morgan Freeman Text-to-Speech

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a Text-to-Speech (TTS) system that generates audio in the voice of Morgan Freeman using a fine-tuned SpeechT5 model with a HiFi-GAN vocoder. It includes a FastAPI backend for processing text inputs and a web-based frontend for user interaction. Users can input text through a sleek web interface and receive generated audio resembling Morgan Freeman's iconic deep voice.

## Features

*   **TTS Generation**: Converts text to speech using a fine-tuned SpeechT5 model trained on Morgan Freeman's voice.
*   **Web Interface**: A responsive, neon-themed frontend for entering text, playing audio, and downloading results.
*   **API Backend**: A FastAPI server to handle TTS requests and serve audio files.
*   **Voice Enhancement**: Includes pitch shifting and bass boosting to emulate Morgan Freeman's deep, resonant tone.
*   **Dataset**: Uses a custom dataset of Morgan Freeman audio clips for fine-tuning.

## Project Structure
```text
MorganFreemanTTS/
├── SoundFiles/
│ ├── metadata_train.json # Training dataset metadata
│ ├── metadata_eval.json # Validation dataset metadata
│ └── Morgan Freeman_*.wav # Audio clips (Pattern representing multiple files)
│
├── frontend/
│ ├── index.html # Main web interface HTML
│ ├── Freeman.png # Image used in the frontend
│ └── serve.py # Simple Python HTTP server for frontend (optional)
│
├── speecht5_finetuned/ # Directory containing the fine-tuned SpeechT5 model files
│
├── morgan13.py # Core Python script for TTS training and generation logic
├── backend.py # FastAPI application script for the backend server
├── avg_speaker_embedding.pt # Pre-computed speaker embedding file for Morgan Freeman's voice
└── README.md # This file


## Prerequisites

### Software
*   **Python**: 3.8 or higher
*   **System Dependencies**:
    *   macOS: `brew install libsndfile ffmpeg`
    *   Linux: `sudo apt-get update && sudo apt-get install libsndfile1 ffmpeg`

### Hardware
*   **GPU**: Recommended for faster training/inference (CUDA support required). CPU can be used but will be significantly slower.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd MorganFreemanTTS
    ```
    *(Replace `<repository-url>` with the actual URL of your repository)*

2.  **Create a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install torch torchaudio transformers fastapi uvicorn python-multipart speechbrain numpy soundfile pydub
    ```
    *(Note: Added `soundfile` and `pydub` as they are often needed for audio manipulation like bass boost)*

4.  **Prepare Dataset:**
    *   Ensure the `SoundFiles/` directory contains your `.wav` files of Morgan Freeman's voice.
    *   Ensure `metadata_train.json` and `metadata_eval.json` exist and contain entries like:
        ```json
        {"audio_filepath": "SoundFiles/Morgan Freeman_00000046.wav", "text": "Sample text here"}
        ```
    *   Audio files should be clean, mono, 16kHz `.wav` files. You can convert other formats using `ffmpeg`:
        ```bash
        ffmpeg -i input.mp3 -ac 1 -ar 16000 SoundFiles/Morgan_Freeman_00000131.wav
        ```

5.  **Verify Model Files:**
    *   Ensure the `speecht5_finetuned/` directory contains the fine-tuned SpeechT5 model components.
    *   Ensure the `avg_speaker_embedding.pt` file exists in the root directory.
    *   If these are missing, you will need to train the model first (see Usage section below).

## Usage

### 1. Training the Model (Optional)

If you don't have a pre-trained model (`speecht5_finetuned/` and `avg_speaker_embedding.pt`) or want to improve an existing one, run the training script:

```bash
python3 morgan13.py
Use code with caution.
You will be prompted to choose an option:
1: Train a new model: Requires a sufficient dataset in SoundFiles/ and corresponding metadata files.
2: Resume training: Continues training from an existing checkpoint in speecht5_finetuned/.
3: Skip training: Proceeds directly to generation (requires pre-trained model files).
For training:
Consider enabling data augmentation in morgan13.py, especially if your dataset is small.
Key training parameters (editable in morgan13.py):
epochs: 100 (or more, depending on dataset size and quality)
batch_size: 4 (adjust based on GPU memory)
learning_rate: 1e-5
2. Running the Backend Server
Start the FastAPI server which handles TTS requests:
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
Use code with caution.
Bash
The API will be available at http://localhost:8000 (or your machine's IP address on port 8000).
You can test the API endpoint directly using curl:
curl -X POST -H "Content-Type: application/json" -d '{"text":"Hello, this is a test."}' http://localhost:8000/generate_speech/ --output test_output.wav
Use code with caution.
Bash
This should save the generated audio as test_output.wav.
3. Running the Frontend
Serve the web interface:
cd frontend
python3 serve.py
Use code with caution.
Bash
Alternatively, if you have Node.js installed, you can use http-server:
# Install http-server globally (if you haven't already)
npm install -g http-server

# Serve the frontend directory with CORS enabled
cd frontend
http-server -p 8080 --cors
Use code with caution.
Bash
Open your web browser and navigate to http://localhost:8080.
4. Generating Audio via Frontend
Enter the text you want to convert into the text area (e.g., "This is Morgan Freeman speaking.").
Click the "Generate Speech" button. The request will be sent to the backend.
Once processing is complete, an audio player will appear.
Use the player controls to listen to the generated audio.
Click the "Download Audio" button to save the audio file locally as morgan-freeman-speech.wav.
Improving Voice Quality
If the generated audio doesn't sound convincingly like Morgan Freeman or has poor quality, consider the following:
Dataset Quality & Quantity:
Ensure SoundFiles/ contains at least 100+ high-quality, clean audio clips of Morgan Freeman speaking clearly. Aim for 1-2 hours of total audio.
Verify that metadata_train.json and metadata_eval.json have accurate transcriptions matching the audio content and correct file paths.
Retraining:
Run python3 morgan13.py and choose option 2 to resume training.
Enable data augmentation if not already enabled.
Train for more epochs (e.g., 100 or more) with a suitable learning rate (e.g., 1e-5). Monitor evaluation loss.
Voice Enhancement Parameters:
Verify that morgan13.py (specifically the generate_speech function, potentially called by the backend) includes appropriate post-processing steps like:
Bass Boost: Using pydub or librosa functions (e.g., apply_bass_boost possibly using effects.equalizer or effects.low_pass_filter if pydub is used, or similar filters with librosa). The provided backend.py likely needs to call this.
Pitch Shift: Applying a downward pitch shift (e.g., -2 semitones) using libraries like librosa or soundfile.
Speaker Embedding:
Ensure avg_speaker_embedding.pt was generated correctly from your dataset. Using a high-quality speaker encoder model like speechbrain/spkrec-ecapa-voxceleb (as likely used in morgan13.py) is crucial. Regenerate the embedding if you significantly change the dataset.
Troubleshooting
CORS Errors (Frontend cannot reach backend):
Ensure the FastAPI backend (backend.py) includes CORS middleware correctly configured (allowing * or the specific frontend origin http://localhost:8080).
Make sure you are serving the frontend using python3 serve.py or http-server --cors, not by opening index.html directly from the file system.
No Audio Generated / Backend Error:
Check the console output where you ran uvicorn backend:app ... for any error messages.
Test the backend directly using the curl command provided in the "Running the Backend" section. If curl fails, the issue is likely in backend.py or the TTS generation logic in morgan13.py.
Verify model files (speecht5_finetuned/, avg_speaker_embedding.pt) exist and are accessible.
Poor Voice Quality (Robotic, Muffled, Incorrect Tone):
Refer to the "Improving Voice Quality" section. This usually points to issues with the dataset size/quality or insufficient training.
Try retraining with more data, more epochs, and data augmentation.
Adjust pitch shift and bass boost parameters if the tone is off.
If you encounter specific issues (e.g., consistently robotic sound), please provide details when seeking help.
Model Files Missing (speecht5_finetuned/ or avg_speaker_embedding.pt):
You need to train the model first. Run python3 morgan13.py and choose option 1 (Train a new model).
Contributing
Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests. Potential areas for contribution include:
Enhancing voice similarity through better datasets, model tuning, or post-processing.
Adding new features to the frontend (e.g., voice controls, preview options).
Optimizing the backend for faster inference speed or lower resource usage.
Improving documentation and setup instructions.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments
Based on the powerful SpeechT5 model from Microsoft.
Utilizes the HiFi-GAN vocoder for high-fidelity audio synthesis.
Backend powered by FastAPI.
Leverages the Transformers library by Hugging Face and potentially SpeechBrain.
Inspired by the iconic and resonant voice of Morgan Freeman.
This version uses the standard box-drawing characters within a Markdown code block, which should render correctly as a visual tree in most modern Markdown viewers (like those on GitHub, GitLab, etc.).