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
        "Esta gravação contém um diálogo entre um professor e dois alunos sobre temas ambientais e de modelagem "
        "atmosférica. Eles discutem o BRAMS (Brazilian developments on the Regional Atmospheric Modeling System), que "
        "aborda previsão e pesquisa em química atmosférica e ciclos bio geoquímicos,o CEMPA(Centros de Monitoramento e "
        "Proteção Ambiental) voltado à conservação de áreas protegidas, o bioma do Cerrado, famoso por sua "
        "biodiversidade e importância para recursos hídricos, e o INPE (Instituto Nacional de Pesquisas Espaciais), "
        "responsável pelo monitoramento de desmatamento e mudanças ambientais via satélite. O professor explica "
        "conceitos e processos, enquanto os alunos fazem perguntas e interagem ativamente com o conteúdo."
    )
}
