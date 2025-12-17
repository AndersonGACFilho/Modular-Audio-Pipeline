# Audio Processing & Transcription Pipeline v2.0

A modular, production-ready Python pipeline for audio transcription with speaker diarization.

## Features

- **Modular Architecture**: Clean separation of concerns with dependency injection
- **Smart Preprocessing**: Auto-detect noise segments, optional vocal separation
- **Timestamp Preservation**: Map processed timestamps back to original audio
- **Checkpoint/Resume**: Resume interrupted processing for long files
- **Lazy Loading**: Load heavy models only when needed
- **Configurable**: JSON configuration files, CLI arguments, or programmatic API
- **Error Handling**: Custom exceptions with detailed error messages
- **Retry Logic**: Exponential backoff for transient failures

## Quick Start

```bash
# Basic usage - process all files in ./files directory
python main.py

# Process specific file
python main.py --input recording.mp3

# Use English transcription
python main.py --language en --model large-v3

# Disable diarization for single speaker
python main.py --no-diarization
```

## Installation

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate AudioPipeline
```

### Using pip

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install openai-whisper pyannote.audio demucs pydub noisereduce pyloudnorm webrtcvad-wheels
```

### System Dependencies

- FFmpeg (required for media conversion)
- CUDA 11.8+ (recommended for GPU acceleration)

## Configuration

### Environment Variables

Create a `.env` file:

```dotenv
HF_TOKEN=your_hugging_face_token_here
```

### JSON Configuration

```json
{
  "media_dir": "./recordings",
  "transcription": {
    "model": "large-v3-turbo",
    "language": "pt",
    "prompt": "Transcreva com precisão..."
  },
  "diarization": {
    "enabled": true,
    "min_speakers": 2,
    "max_speakers": 4
  }
}
```

### CLI Options

```
Input Options:
  --media-dir, -d       Directory containing media files
  --input, -i           Specific input file to process
  --config, -c          Path to JSON configuration file

Transcription Options:
  --model, -m           Whisper model (tiny/base/small/medium/large/large-v3-turbo)
  --language, -l        Language code (pt, en, es, etc.)
  --prompt, -p          Custom transcription prompt
  --prompt-preset       Use preset prompt (pt_instructions, en_general, etc.)

Processing Options:
  --separate-vocals     Enable vocal separation for audio with music
  --auto-separate       Auto-detect if vocal separation is needed
  --no-diarization      Disable speaker diarization
  --no-vad              Disable Voice Activity Detection
  --no-noise-reduction  Disable noise reduction
  --min-speakers        Minimum expected speakers
  --max-speakers        Maximum expected speakers

Output Options:
  --output-dir, -o      Directory for output files
  --preserve-timestamps Preserve original timestamps (default: True)
```

## Programmatic Usage

### Basic Usage

```python
from audio_pipeline import AudioPipeline, PipelineConfig

config = PipelineConfig(media_dir="./recordings")
pipeline = AudioPipeline(config)
result = pipeline.run()

if result.success:
    print(f"Output: {result.output_file}")
    for seg in result.segments:
        print(f"[{seg['speaker']}] {seg['text']}")
```

### Dependency Injection

```python
from audio_pipeline import (
    AudioPipeline,
    PipelineConfig,
    WhisperTranscriber,
    FasterWhisperTranscriber,  # Alternative implementation
    NoOpDiarizer,  # Disable diarization
)

# Use custom components
config = PipelineConfig()
transcriber = FasterWhisperTranscriber(model_name="large-v3")
diarizer = NoOpDiarizer()  # Skip diarization

pipeline = AudioPipeline(
    config=config,
    transcriber=transcriber,
    diarizer=diarizer
)
```

### Custom Configuration

```python
from audio_pipeline import PipelineConfig

config = PipelineConfig(
    media_dir="./recordings",
    lazy_load_models=True,
    checkpoint_enabled=True,
)

# Customize transcription
config.transcription.model = "large-v3-turbo"
config.transcription.language = "pt"
config.transcription.prompt = "Reunião de trabalho..."

# Disable vocal separation
config.vocal_separation.enabled = False

# Configure diarization
config.diarization.min_speakers = 2
config.diarization.max_speakers = 5

# Validate configuration
config.validate()
```

## Architecture

### Pipeline Flow

```
Media Input → Convert to WAV → Denoise → [Vocal Separation] → Normalize
    → [VAD] → Transcribe → Diarize → Align → Remove Redundancy → Output JSON
```

### Module Responsibilities

| Module | Description |
|--------|-------------|
| `media_handler.py` | File discovery, validation, FFmpeg conversion |
| `preprocessor.py` | Noise reduction, normalization, silence removal |
| `separator.py` | Demucs vocal separation with auto-detection |
| `vad.py` | WebRTC Voice Activity Detection |
| `transcriber.py` | Whisper ASR with lazy loading |
| `diarizer.py` | Pyannote speaker diarization |
| `redundancy.py` | Duplicate segment filtering |
| `pipeline.py` | Orchestration with DI support |

### Protocols (Interfaces)

All components implement protocols for easy swapping:

```python
class TranscriberProtocol(Protocol):
    def transcribe(self, input_wav: str) -> Dict[str, Any]: ...
    def is_loaded(self) -> bool: ...
    def load_model(self) -> None: ...
```

## Key Improvements in v2.0

### 1. Dependency Injection

Replace any component without modifying pipeline code:

```python
class MyCustomTranscriber(TranscriberProtocol):
    def transcribe(self, input_wav: str) -> dict:
        # Your implementation
        pass

pipeline = AudioPipeline(transcriber=MyCustomTranscriber())
```

### 2. Auto Noise Detection

Instead of using fixed first 0.5s, automatically finds silent segments:

```python
config.noise_reduction.auto_detect_noise = True
```

### 3. Timestamp Preservation

Map processed timestamps back to original audio:

```python
# Output segment includes both processed and original timestamps
{
    "speaker": "SPEAKER_01",
    "start": 10.5,          # In processed audio
    "end": 15.2,
    "original_start": 12.3, # In original audio
    "original_end": 17.8,
    "text": "..."
}
```

### 4. Checkpoint/Resume

Resume interrupted processing:

```python
config.checkpoint_enabled = True
# If processing fails at step 5, restart will skip completed steps
```

### 5. Lazy Model Loading

Models load only when needed:

```python
config.lazy_load_models = True
# Whisper loads on first transcribe(), not at pipeline init
```

### 6. Custom Exceptions

Specific exception types for better error handling:

```python
from audio_pipeline import (
    MediaNotFoundError,
    TranscriptionError,
    DiarizationError,
)

try:
    result = pipeline.run()
except MediaNotFoundError:
    print("No audio files found")
except TranscriptionError as e:
    print(f"Transcription failed: {e.details}")
```

### 7. Retry Logic

Automatic retry with exponential backoff:

```python
config.retry.max_attempts = 3
config.retry.initial_delay_s = 1.0
config.retry.exponential_backoff = True
```

## Output Format

```json
{
  "metadata": {
    "source_file": "/path/to/recording.mp3",
    "config": {
      "model": "large-v3-turbo",
      "language": "pt"
    }
  },
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 5.2,
      "text": "Bom dia, vamos começar a reunião.",
      "original_start": 0.0,
      "original_end": 5.5
    },
    {
      "speaker": "SPEAKER_01",
      "start": 5.4,
      "end": 12.1,
      "text": "Boa, temos três itens na pauta hoje.",
      "original_start": 5.8,
      "original_end": 12.9
    }
  ]
}
```

## Supported Formats

### Audio
- MP3, M4A, WAV, OGG, FLAC, AAC, WMA, OPUS

### Video (extracts audio)
- MP4, AVI, MOV, WMV, MKV, WebM

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for 10x faster processing
2. **Choose Right Model**: `large-v3-turbo` balances quality and speed
3. **Disable Unused Features**: Skip diarization for single-speaker audio
4. **Use Checkpoints**: Enable for files >30 minutes

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Implement with tests
4. Submit pull request

## License

MIT License - see LICENSE file for details.
