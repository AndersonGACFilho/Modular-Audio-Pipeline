# Teste se tudo está OK
from audio_pipeline import AudioPipeline, PipelineConfig
config = PipelineConfig.from_json('config.json')
pipeline = AudioPipeline(config)
print('✓ Pipeline OK')

if pipeline.llm_processor:
    info = pipeline.llm_processor.get_backend_info()
    print(f'✓ LLM: {info["backend"]} ({info["model"]})')

print('🎉 Tudo funcionando!')