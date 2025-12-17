from pathlib import Path

CONFIG = {
    "sample_rate": 16000,
    "vad_mode": 1,
    "frame_duration_ms": 30,
    "padding_duration_ms": 500,
    "start_threshold": 0.5,
    "stop_threshold": 0.9,
    "model": "large-v3-turbo",
    "language": "pt",
    "response_format": "verbose_json",
    "media_dir": str(Path("./files").resolve()),
    "redundancy_threshold": 0.85,
    "prompt": (
        "Esta gravação é meu chefe me dando instruções sobre tarefas de trabalho. "
        "Transcreva o áudio em português de forma precisa, mantendo a pontuação correta "
        "e indicando claramente quaisquer pausas ou hesitações. "
        "Formate a transcrição em parágrafos para melhor legibilidade."
    )
}
