Morgan Freeman Text-to-Speech

This project implements a Text-to-Speech (TTS) system that generates audio in the voice of Morgan Freeman using a fine-tuned SpeechT5 model with a HiFi-GAN vocoder. The system includes a FastAPI backend for processing text inputs and a web-based frontend for user interaction. Users can input text through a sleek web interface and receive generated audio resembling Morgan Freeman's iconic deep voice.

Features





TTS Generation: Converts text to speech using a fine-tuned SpeechT5 model trained on Morgan Freeman's voice.



Web Interface: A responsive, neon-themed frontend for entering text, playing audio, and downloading results.



API Backend: A FastAPI server to handle TTS requests and serve audio files.



Voice Enhancement: Includes pitch shifting and bass boosting to emulate Morgan Freeman's deep, resonant tone.



Dataset: Uses a custom dataset of Morgan Freeman audio clips for fine-tuning.

Project Structure

MorganFreemanTTS/
├── SoundFiles/                 # Directory containing audio files and metadata
│   ├── metadata_train.json     # Training dataset metadata
│   ├── metadata_eval.json      # Validation dataset metadata
│   └── Morgan Freeman_*.wav    # Audio clips
├── frontend/                   # Web frontend files
│   ├── index.html              # Main web interface
│   ├── Freeman.png             # Morgan Freeman image
│   └── serve.py                # Optional Python script to serve frontend
├── speecht5_finetuned/         # Fine-tuned SpeechT5 model
├── morgan13.py                 # Core TTS script (training and generation)
├── backend.py                  # FastAPI backend server
├── avg_speaker_embedding.pt    # Speaker embedding for Morgan Freeman
└── README.md                   # This file

Prerequisites





Python: 3.8 or higher



System Dependencies:





On macOS: brew install libsndfile ffmpeg



On Linux: apt-get install libsndfile1 ffmpeg



Hardware: GPU recommended for faster training/inference (CUDA support required).

Installation





Clone the Repository:

git clone <repository-url>
cd MorganFreemanTTS



Create a Virtual Environment:

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install Python Dependencies:

pip install torch torchaudio transformers fastapi uvicorn python-multipart speechbrain numpy



Prepare Dataset:





Ensure SoundFiles/ contains WAV files of Morgan Freeman's voice and metadata_train.json/metadata_eval.json with entries like:

{"audio_filepath": "SoundFiles/Morgan Freeman_00000046.wav", "text": "Sample text here"}



Audio files should be clean, mono, 16kHz WAVs. Convert if needed:

ffmpeg -i input.mp3 -ac 1 -ar 16000 SoundFiles/Morgan_Freeman_00000131.wav



Verify Model:





Ensure speecht5_finetuned/ contains the fine-tuned SpeechT5 model and avg_speaker_embedding.pt exists.



If not, train the model (see Usage below).

Usage

Training the Model





Run the TTS script to fine-tune or resume training:

python3 morgan13.py



Choose an option:





1: Train a new model (requires sufficient dataset).



2: Resume training from speecht5_finetuned/.



3: Skip training and generate audio (requires pre-trained model).



For training, enable data augmentation for better results (recommended for small datasets).



Training parameters (editable in morgan13.py):





Epochs: 100



Batch Size: 4



Learning Rate: 1e-5

Running the Backend





Start the FastAPI server:

uvicorn backend:app --reload



The API will be available at http://localhost:8000. Test with:

curl -X POST -H "Content-Type: application/json" -d '{"text":"Hello, this is Morgan Freeman."}' http://localhost:8000/generate_speech/ --output test.wav

Running the Frontend





Serve the frontend:

cd frontend
python3 serve.py

Or use Node.js http-server:

npm install -g http-server
http-server -p 8080 --cors



Open http://localhost:8080 in a browser.



Enter text, click "Generate Speech," and play/download the audio.

Generating Audio via Frontend





Input text in the textarea (e.g., “This is Morgan Freeman speaking.”).



Click “Generate Speech” to produce audio.



Use the audio player to listen or click “Download Audio” to save morgan-freeman-speech.wav.

Improving Voice Quality

If the audio doesn’t sound like Morgan Freeman or has poor quality:





Dataset:





Ensure SoundFiles/ has 100+ clean, high-quality Morgan Freeman clips (1-2 hours total).



Check metadata_train.json for correct file paths and transcriptions.



Retrain:





Resume training with:

python3 morgan13.py





Choose option 2, enable augmentation, and use 100 epochs with learning_rate=1e-5.



Voice Enhancement:





Verify morgan13.py includes bass boost (apply_bass_boost with equalizer_biquad) and pitch shift (-2 steps) in generate_speech.



Speaker Embedding:





Use speechbrain/spkrec-ecapa-voxceleb for a more accurate embedding (see morgan13.py for implementation).

Troubleshooting





CORS Errors:





Ensure backend.py includes CORS middleware.



Serve the frontend with serve.py or http-server instead of opening index.html directly.



No Audio:





Check backend logs (uvicorn output) for errors.



Test API directly with curl (see above).



Poor Voice Quality:





Verify dataset size and quality.



Retrain with more epochs and augmentation.



Share issues (e.g., robotic, muffled) for targeted fixes.



Model Missing:





If speecht5_finetuned/ or avg_speaker_embedding.pt is missing, train the model with option 1.

Contributing

Contributions are welcome! To improve the project:





Enhance voice quality with better datasets or models.



Add features to the frontend (e.g., voice preview, pitch adjustment).



Optimize backend performance for faster inference.

License

This project is licensed under the MIT License.

Acknowledgments





Built with SpeechT5 and HiFi-GAN.



Powered by FastAPI and Transformers.



Inspired by Morgan Freeman’s iconic voice.

## Authors

- Tech Lead - [Yevgeniy Kim](https://github.com/musicaleugene)
- Software Developer - [Brynn Williams](https://github.com/bgbranfl)
- Software Developer - Jake Land
- Software Developer - [Thomas Hughes](https://github.com/7itanium)
- Scrum Master/Tester - Tabarek Ibrahim
- Information Security Officer - DeJuan Leffall



