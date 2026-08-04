# VideoTranscripter

VideoTranscripter is an audio and video transcription pipeline with speaker
diarization, optional LLM analysis, local GPU processing, asynchronous MongoDB
job tracking, and RabbitMQ-based background processing.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Local setup](#local-setup)
- [CLI usage](#cli-usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Asynchronous jobs](#asynchronous-jobs)
- [Docker deployment](#docker-deployment)
- [Media lifecycle](#media-lifecycle)
- [Development](#development)
- [Current limitations](#current-limitations)

## Features

- Audio and video discovery with deterministic local queue processing.
- Audio conversion, denoising, normalization, silence removal, and VAD.
- Faster-Whisper transcription with language, hotword, and timestamp options.
- Pyannote speaker diarization and speaker/transcript alignment.
- Optional LLM analysis with profile routing and bounded chunk processing.
- A 16-step terminal progress bar with current file, queue position, stage, and
  LLM chunk details.
- Hexagonal architecture with application ports and infrastructure adapters.
- Asynchronous MongoDB persistence through PyMongo's `AsyncMongoClient`.
- Durable RabbitMQ publishing, consumption, and dead-letter routing through
  `aio-pika`.

## Requirements

- Python 3.11.
- FFmpeg available on `PATH`.
- NVIDIA driver and CUDA-capable GPU for accelerated processing (recommended).
- Docker with NVIDIA Container Toolkit for GPU-enabled container execution
  (optional).

## Local setup

Install [uv](https://docs.astral.sh/uv/) if it is not already available, then
create the project environment:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync --extra dev
```

Create a local `.env` file for secrets. Do not commit it.

```dotenv
HF_TOKEN=your_hugging_face_token
```

## CLI usage

Without `--input`, the CLI discovers and processes every supported media file
in the configured media directory. Audio files are processed before video files;
files within each group are sorted alphabetically.

```powershell
# Process all supported files in the directory configured by config.json.
uv run python app/src/audio_pipeline/entrypoints/cli.py -c config.json

# Process one file only.
uv run python app/src/audio_pipeline/entrypoints/cli.py -c config.json --input recording.mp4
```

When attached to an interactive terminal, progress remains below log output:

```text
File 2/7: daily.mp4 - Transcription (9/16)
```

The LLM stage also reports its substeps, such as `chunk 1/3` and
`consolidating 3 chunks`. Progress rendering is disabled for redirected output
and non-interactive execution.

## Configuration

`config.json` contains CLI defaults. The RabbitMQ worker receives operational
configuration from environment variables and semantic processing options from
the persisted `AudioJob`.

For the local CLI, place queued audio and video in `files/incoming`. This is
the default configured by `config.json` and matches Docker's `/data/incoming`.
Use `--media-dir <folder>` only when intentionally processing another folder.

### Transcription configuration

```json
{
  "transcription": {
    "backend": "faster-whisper",
    "model": "large-v3",
    "language": "pt",
    "locale": "pt-BR",
    "initial_prompt": null,
    "hotwords": ["SIGOR", "MTR", "CNPJ", "eControle"],
    "condition_on_previous_text": false
  }
}
```

Use `hotwords` for domain vocabulary. Do not put behavioral instructions such
as “summarize this meeting” or “extract decisions” in `initial_prompt`: an ASR
model can reproduce those instructions as spoken text. Put those instructions
in the LLM analysis profile instead.

### Local LLM fallback

Ollama is the preferred LLM backend. The local Hugging Face model is used only
when Ollama and OpenAI are unavailable. Its default fallback settings use SDPA,
greedy generation, and the KV cache to preserve VRAM and avoid long generation
times on 8 GB GPUs:

```json
{
  "local_max_new_tokens": 384,
  "local_attention_implementation": "sdpa"
}
```

If a model does not support SDPA, the pipeline logs the reason and retries with
`eager` attention rather than failing the job.

### Runtime preflight

Before any media is processed, the CLI logs checks for the media/results
directories, FFmpeg, CUDA/GPU VRAM, the Hugging Face token, and the configured
Ollama host/model. A failed Ollama check is explicit in the log before the
pipeline can choose its local fallback.

## Fine-tuning dataset ETL

Trusted `llm_analysis` fields from final transcription results can be exported
as chat-format JSONL for supervised fine-tuning. The exporter excludes partial
snapshots, incomplete analyses, short transcripts, and duplicate examples. It
also assigns each source meeting deterministically to a train, validation, or
test split.

```powershell
$env:PYTHONPATH = "app/src"
.venv\Scripts\python.exe -m audio_pipeline.finetuning `
  --input-dir files/results `
  --output-dir files/finetuning
```

The command writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and
`manifest.json`. It prepares data only; LoRA/QLoRA training is intentionally a
separate execution step so its model and hardware configuration can be chosen
explicitly.

### Worker environment variables

| Variable | Purpose | Default |
|---|---|---|
| `AUDIO_PIPELINE_DATA_ROOT` | Shared media and processing root | `data` |
| `AUDIO_PIPELINE_MONGODB_URI` | MongoDB connection URI | Local MongoDB URI |
| `AUDIO_PIPELINE_RABBITMQ_URL` | RabbitMQ connection URI | Local RabbitMQ URI |
| `AUDIO_PIPELINE_JOB_LEASE_SECONDS` | Processing lease duration | `3600` |
| `HF_TOKEN` | Hugging Face token for diarization | Empty |

## Architecture

```text
CLI or future web application
        |
        +-- SubmitAudioJob --> JobRepository --> MongoDB
        |                          |
        |                          +-- JobPublisher --> RabbitMQ { job_id }
        |
        +-- Local CLI --> AudioPipeline

RabbitMQ worker
        |
        +-- ProcessAudioJob --> AudioProcessor --> AudioPipeline
                                               |
                                               +-- FFmpeg and media processing
                                               +-- Faster-Whisper
                                               +-- Pyannote
                                               +-- Optional LLM analysis
```

| Layer | Responsibility |
|---|---|
| `domain/` | `AudioJob`, lifecycle states, result/error objects, immutable options. |
| `application/use_cases/` | Job submission and processing orchestration. |
| `application/ports/` | Repository, publisher, storage, and processor contracts. |
| `bootstrap/` | Composition root for concrete adapters. |
| `infrastructure/` | MongoDB, RabbitMQ, storage, media processing, ASR, diarization, and LLM adapters. |
| `entrypoints/` | CLI and RabbitMQ worker processes. |

### Job lifecycle

```text
queued --claim--> processing --success--> completed
                         |
                         +--failure--> failed
```

MongoDB claims jobs atomically. Completion and failure are aggregate transitions
performed by `AudioJob` and persisted through the repository.

## Asynchronous jobs

`SubmitAudioJob` and `ProcessAudioJob` are asynchronous use cases. MongoDB uses
PyMongo's `AsyncMongoClient`; RabbitMQ uses `aio-pika` in the same event loop.

RabbitMQ carries a minimal, durable message:

```json
{
  "schema_version": 1,
  "job_id": "uuid"
}
```

MongoDB remains the source of truth for source metadata, checksums,
transcription and analysis options, state, attempts, errors, and result
references. The message intentionally does not duplicate web application data.

## Docker deployment

The Compose stack runs three services:

- `mongo` for job and status persistence.
- `rabbitmq` for durable work queues and dead-letter routing.
- `audio-pipeline`, a single image that runs either the CLI or the RabbitMQ
  consumer.

```powershell
Copy-Item compose.env.example .env
# Edit .env and replace every example password.
docker compose up --build
```

`AUDIO_PIPELINE_MODE` selects the entrypoint of the `audio-pipeline` service:

```dotenv
# Default: consume RabbitMQ jobs.
AUDIO_PIPELINE_MODE=rabbit

# Process files from /data/incoming with the CLI.
# AUDIO_PIPELINE_MODE=cli
```

### GPU workload profile

Set `AUDIO_PIPELINE_GPU_PROFILE` in `.env` to reduce GPU contention. The
profile changes inference batch sizes; it does not impose a hard percentage or
VRAM quota, but lower batches leave more GPU time available to interactive
Windows applications.

| Profile | Transcription batch | Diarization batches | Recommended use |
|---|---:|---:|---|
| `fast` | 4 | 32 / 32 | Dedicated processing machine |
| `balanced` | 2 | 16 / 16 | Default Docker workload |
| `background` | 1 | 4 / 4 | Use the computer while processing |

Restart the `audio-pipeline` service after changing the profile.

### Silero VAD cache

The Docker image downloads and caches the Silero VAD implementation while it
is built. Processing a file therefore does not depend on GitHub being
available. After updating this project, rebuild before starting a new job:

```powershell
docker compose build audio-pipeline
docker compose up -d audio-pipeline
```

RabbitMQ Management is exposed at `http://localhost:15673` by default. Set
`RABBITMQ_MANAGEMENT_PORT` in `.env` to use another available host port.

The local `files` directory is bind-mounted at `/data`. Files written locally
under `files/incoming` are immediately available to the container at
`/data/incoming`; results and temporary artifacts written by the container are
immediately available in the same local `files` directory. The worker can then
process the source without sending media bytes through RabbitMQ.

## Media lifecycle

```text
/data/incoming/<job>/source.mp4
        |
        +-- /data/processing/<job>/...      temporary artifacts and checkpoints
        |
        +-- /data/results/YYYY/MM/DD/<job>/ current JSON output
```

Converted WAV, denoised, normalized, and VAD audio are processing artifacts.
The transcription JSON and its metadata are the intended product output.

## Transcription output

```json
{
  "metadata": {
    "source_file": "/data/incoming/.../source.mp4",
    "config": {
      "model": "large-v3",
      "language": "pt"
    }
  },
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 5.2,
      "original_start": 0.0,
      "original_end": 5.5,
      "text": "Transcribed text"
    }
  ],
  "llm_analysis": {}
}
```

## Development

```powershell
uv run --extra dev pytest -q
uv run python -m compileall -q app/src
docker compose --env-file compose.env.example config
```

## Supported formats

- Audio: MP3, M4A, WAV, OGG, FLAC, AAC, WMA, and OPUS.
- Video: MP4, AVI, MOV, WMV, MKV, WebM, and M4V.

## Current limitations

- No HTTP or web service is included yet. A future web application should write
  uploads to the shared volume and call `SubmitAudioJob`.
- The final transcription JSON is currently written to `results/`. Persisting it
  in a dedicated MongoDB collection is the next step for a filesystem-free
  final output.
