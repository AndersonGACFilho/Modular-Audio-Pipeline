from __future__ import annotations

import json
import shutil
import logging
import argparse
import os
import subprocess
import contextlib
import collections
import wave
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub.hf_api import HfFolder
import huggingface_hub
import numpy as np
import openai
import whisper
import webrtcvad
import pyloudnorm as pyln
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.silence import detect_nonsilent
from spleeter.separator import Separator
from pyannote.audio import Pipeline

# Carrega variáveis de ambiente de .env
from dotenv import load_dotenv

# Configure global logging
logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class ProcessadorAudio:
    """
    Pipeline modular para processamento e transcrição de áudio.
    """

    def __init__(
            self,
            media_dir: str,
            sample_rate: int = 16000,
            vad_mode: int = 1,
            frame_duration_ms: int = 30,
            padding_duration_ms: int = 500,
            start_threshold: float = 0.5,
            stop_threshold: float = 0.9,
            model: str = "large-v3-turbo",
            language: str = "pt",
            response_format: str = "verbose_json"
    ):
        logger.info("Inicializando ProcessadorAudio")
        self.media_dir = os.path.abspath(media_dir)
        self.sample_rate = sample_rate
        self.vad_mode = vad_mode
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold
        self.model = model
        self.language = language
        self.response_format = response_format

        # Diretórios de trabalho
        self.temp_dir = os.path.join(self.media_dir, 'temp')
        self.results_dir = os.path.join(self.media_dir, 'results')
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Inicializa separador de stems do Spleeter
        logger.info("Inicializando Separator do Spleeter")
        self.separator = Separator('spleeter:2stems')

    def find_media_file(self) -> tuple[str, bool] | None:
        logger.info("Procurando arquivos de mídia em %s", self.media_dir)
        audio_exts = {'.mp3', '.m4a', '.wav', '.ogg', '.flac', '.aac'}
        video_exts = {'.mp4', '.avi', '.mov', '.wmv'}

        # Remove diretório temporário se existir
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            # Cria novamente o diretório temporário
            os.makedirs(self.temp_dir, exist_ok=True)

        for fname in os.listdir(self.media_dir):
            full = os.path.join(self.media_dir, fname)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in audio_exts:
                logger.info("Arquivo de áudio encontrado: %s", full)
                return full, False
            if ext in video_exts:
                logger.info("Arquivo de vídeo encontrado: %s", full)
                return full, True

        logger.error("Nenhum arquivo de mídia válido encontrado.")
        return None

    def convert_to_wav(self, input_path: str) -> str:
        logger.info("Convertendo %s para WAV mono %dHz", input_path, self.sample_rate)
        base = Path(input_path).stem
        out_path = os.path.join(self.temp_dir, f"{base}_{self.sample_rate}Hz_temp.wav")
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', str(self.sample_rate),
            out_path
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            err = result.stderr.decode(errors='ignore')
            logger.error("Erro na conversão para WAV: %s", err)
            raise RuntimeError(f'Erro na conversão para WAV: {err}')
        logger.info("Conversão concluída: %s", out_path)
        return out_path

    def remove_silence(
            self,
            input_wav: str,
            min_silence_len: int = 250,
            silence_margin: int = 100,
            silence_offset_db: float = 40.0
    ) -> str:
        """
        Remove trechos de silêncio do WAV salvando em self.temp_dir.
        Usa limiar relativo e mantém uma margem de silêncio.
        """
        seg = AudioSegment.from_wav(input_wav)

        # 1) Limiar dinâmico: x dB abaixo da média do áudio
        silence_thresh = seg.dBFS - silence_offset_db

        # 2) Detecta trechos não silenciosos
        nonsilent = detect_nonsilent(
            seg,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )

        cleaned = AudioSegment.empty()
        for start_ms, end_ms in nonsilent:
            # 3) Ajusta margens e evita valores fora de faixa
            start = max(0, start_ms - silence_margin)
            end = min(len(seg), end_ms + silence_margin)
            chunk = seg[start:end]

            # 4) Suaviza a junção com crossfade curto
            if len(cleaned) == 0:
                cleaned = chunk
            else:
                cleaned = cleaned.append(chunk, crossfade=20)

        base = Path(input_wav).stem
        out_path = os.path.join(self.temp_dir, f"{base}_nosilence.wav")
        cleaned.export(out_path, format='wav')
        logger.info("Silêncio removido salvo em: %s", out_path)
        return out_path

    def read_wave(self, path: str) -> tuple[bytes, int]:
        logger.info("Lendo WAV: %s", path)
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            if wf.getsampwidth() != 2:
                raise ValueError('Formato inválido: espera 16-bit PCM')
            sr = wf.getframerate()
            if sr not in (8000, 16000, 32000, 48000):
                raise ValueError('Taxa de amostragem inválida')
            if wf.getnchannels() != 1:
                raise ValueError('Áudio precisa ser mono')
            pcm = wf.readframes(wf.getnframes())
        logger.info("WAV lido: %d bytes a %dHz", len(pcm), sr)
        return pcm, sr

    def write_wave(self, path: str, audio: bytes, sample_rate: int) -> None:
        logger.info("Escrevendo WAV: %s", path)
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

    def reduce_stationary_noise(self, input_wav: str) -> str:
        logger.info("Reduzindo ruído estacionário em: %s", input_wav)
        pcm, sr = self.read_wave(input_wav)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        noise_clip = audio[:int(sr * 0.5)]
        reduced = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip, prop_decrease=1.0)

        base = Path(input_wav).stem
        out = os.path.join(self.temp_dir, f"{base}_denoised.wav")
        self.write_wave(out, reduced.astype(np.int16).tobytes(), sr)
        logger.info("Ruído reduzido salvo em: %s", out)
        return out

    def extract_vocals(self, input_wav: str, chunk_minutes: float = 5.0) -> str:
        logger.info("Extraindo vocais de: %s", input_wav)
        seg = AudioSegment.from_wav(input_wav)
        total_ms = len(seg)
        chunk_ms = int(chunk_minutes * 60 * 1000)
        vocals_full = AudioSegment.silent(duration=0, frame_rate=self.sample_rate)

        separator = Separator('spleeter:2stems', multiprocess=False)
        for start_ms in range(0, total_ms, chunk_ms):
            end_ms = min(start_ms + chunk_ms, total_ms)
            chunk = seg[start_ms:end_ms]
            chunk_fn = f"chunk_{start_ms//1000}_{end_ms//1000}.wav"
            chunk_path = os.path.join(self.temp_dir, chunk_fn)
            chunk.export(chunk_path, format='wav')
            logger.info("Processando chunk: %s", chunk_fn)

            out_dir = os.path.join(self.temp_dir, 'stems')
            os.makedirs(out_dir, exist_ok=True)
            separator.separate_to_file(chunk_path, out_dir)

            stem_dir = os.path.join(out_dir, Path(chunk_fn).stem)
            vocals_path = os.path.join(stem_dir, 'vocals.wav')
            vocals_seg = AudioSegment.from_wav(vocals_path)
            vocals_full += vocals_seg

            os.remove(chunk_path)
            shutil.rmtree(stem_dir)

        base = Path(input_wav).stem
        out_path = os.path.join(self.temp_dir, f"{base}_vocals_full.wav")
        vocals_full.export(out_path, format='wav')
        logger.info("Vocais extraídos salvos em: %s", out_path)
        return out_path

    def normalize_audio(self, input_wav: str) -> str:
        logger.info("Normalizando áudio (pydub) em: %s", input_wav)
        seg = AudioSegment.from_wav(input_wav)
        norm_seg = normalize(seg)
        norm_seg = norm_seg.set_frame_rate(self.sample_rate).set_channels(1)
        base = Path(input_wav).stem
        out = os.path.join(self.temp_dir, f"{base}_norm.wav")
        norm_seg.export(out, format='wav')
        logger.info("Áudio normalizado salvo em: %s", out)
        return out

    def normalize_loudness(self, input_wav: str, target_lufs: float = -16.0) -> str:
        logger.info("Normalizando loudness em: %s para %0.1f LUFS", input_wav, target_lufs)
        seg = AudioSegment.from_wav(input_wav)
        samples_int16 = np.array(seg.get_array_of_samples(), dtype=np.int16)
        samples = samples_int16.astype(np.float32) / 32768.0

        meter = pyln.Meter(self.sample_rate)
        loudness = meter.integrated_loudness(samples)
        normalized = pyln.normalize.loudness(samples, loudness, target_lufs)

        # Evita clipping: escalona se exceder ±1.0
        peak = np.abs(normalized).max()
        if peak > 1.0:
            normalized = normalized / peak

        out_int16 = np.clip(normalized * 32768.0, -32768, 32767).astype(np.int16)
        out_bytes = out_int16.tobytes()

        base = Path(input_wav).stem
        out_path = os.path.join(self.temp_dir, f"{base}_loudnorm.wav")
        self.write_wave(out_path, out_bytes, self.sample_rate)
        logger.info("Loudness normalizado salvo em: %s", out_path)
        return out_path

    def equalize_speakers(
            self,
            input_wav: str,
            target_lufs: float = -16.0,
            percentile: float = 90.0
    ) -> str:
        if not Path(input_wav).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {input_wav}")

        logger.info("Equalizando locutores em: %s", input_wav)
        full_seg = AudioSegment.from_wav(input_wav)

        # 1) Diarização
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(device)
        diarization = pipeline(input_wav, min_speakers=2, max_speakers=5)
        turns = list(diarization.itertracks(yield_label=True))
        speakers = sorted({spk for _, _, spk in turns})
        logger.info("Locutores detectados: %s", speakers)

        # 2) Medir loudness por locutor em blocos
        sr = full_seg.frame_rate
        meter = pyln.Meter(sr)
        block_samples = int(meter.block_size * sr)
        levels = {spk: [] for spk in speakers}

        for turn, _, spk in turns:
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            seg = full_seg[start_ms:end_ms]
            samples = np.array(seg.get_array_of_samples(), np.int16).astype(np.float32) / 32768.0
            if seg.channels > 1:
                samples = samples.reshape(-1, seg.channels).mean(axis=1)
            # pad if too short
            if samples.size < block_samples:
                samples = np.pad(samples, (0, block_samples - samples.size), 'constant')

            loudness = meter.integrated_loudness(samples)
            if np.isfinite(loudness):
                levels[spk].append(loudness)
            else:
                logger.warning("Loudness inválido para %s no trecho %d-%dms", spk, start_ms, end_ms)

        # 3) Calcular nível de referência por locutor usando percentil
        ref_levels = {}
        for spk, vals in levels.items():
            if vals:
                ref = float(np.percentile(vals, percentile))
            else:
                logger.warning("Sem dados de loudness para %s, usando target %d LUFS", spk, target_lufs)
                ref = target_lufs
            ref_levels[spk] = ref
        # Ganho necessário para trazer ao alvo (limitado a um máximo)
        max_gain_db = 10.0  # você pode ajustar esse valor
        gains = {
            spk: min(target_lufs - lvl, max_gain_db)
            for spk, lvl in ref_levels.items()
        }
        logger.info("Ganho a aplicar por locutor (dB): %s", gains)

        # 4) Reconstrói saída aplicando ganho e overlay
        output = AudioSegment.silent(duration=len(full_seg), frame_rate=sr)
        for turn, _, spk in turns:
            start_ms = int(turn.start * 1000)
            end_ms = int(turn.end * 1000)
            seg = full_seg[start_ms:end_ms]
            adjusted = seg.apply_gain(gains[spk])
            output = output.overlay(adjusted, position=start_ms)

        # 5) Normalização final de loudness
        # Exporta o áudio temporariamente para aplicar a normalização com pyloudnorm
        temp_path = os.path.join(self.temp_dir, "temp_output.wav")
        output.export(temp_path, format='wav')

        # Carrega o áudio com soundfile
        data, rate = sf.read(temp_path)
        meter = pyln.Meter(rate)  # cria o medidor BS.1770
        loudness = meter.integrated_loudness(data)
        normalized_audio = pyln.normalize.loudness(data, loudness, target_lufs)

        # Evita clipping: escalona se exceder ±1.0
        peak = np.abs(normalized_audio).max()
        if peak > 1.0:
            normalized_audio = normalized_audio / peak

        # Salva o áudio normalizado
        out_name = f"{Path(input_wav).stem}_speaker_eq.wav"
        out_path = os.path.join(self.temp_dir, out_name)
        sf.write(out_path, normalized_audio, rate)
        logger.info("Equalização e normalização concluídas: %s", out_path)

        # Remove o arquivo temporário
        os.remove(temp_path)

        return out_path

    def vad_filter(self, input_wav: str, file_name: str) -> str:
        logger.info("Aplicando VAD em: %s", input_wav)
        pcm, sr = self.read_wave(input_wav)
        vad = webrtcvad.Vad(self.vad_mode)

        frame_len = int(sr * (self.frame_duration_ms / 1000.0) * 2)
        frames, offset = [], 0
        while offset + frame_len <= len(pcm):
            frames.append(pcm[offset:offset + frame_len])
            offset += frame_len

        voiced, ring, triggered = [], collections.deque(maxlen=int(self.padding_duration_ms / self.frame_duration_ms)), False
        for chunk in frames:
            is_speech = vad.is_speech(chunk, sr)
            if not triggered:
                ring.append((chunk, is_speech))
                if sum(1 for _, s in ring if s) > self.start_threshold * ring.maxlen:
                    triggered = True
                    voiced.extend(b for b, _ in ring)
                    ring.clear()
            else:
                voiced.append(chunk)
                ring.append((chunk, is_speech))
                if sum(1 for _, s in ring if not s) > self.stop_threshold * ring.maxlen:
                    triggered = False
                    ring.clear()

        out = os.path.join(self.results_dir, f"{file_name}_voz.wav")
        self.write_wave(out, b''.join(voiced), sr)
        logger.info("VAD concluído: %s", out)
        return out

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def transcrever_audio(self, input_wav: str) -> str:
        logger.info("Iniciando transcrição: %s", input_wav)

        model = whisper.load_model(self.model)

        # Refined prompt with detailed context
        prompt = (
            "This recording features a discussion among three individuals: a teacher and two students. "
            "They are conversing about the BRAMS model, CEMPA, the Cerrado biome, and INPE. "
            "BRAMS (Brazilian developments on the Regional Atmospheric Modeling System) is a numerical modeling system designed for regional-scale atmospheric forecasting and research, focusing on atmospheric chemistry, air quality, and biogeochemical cycles. "
            "CEMPA refers to environmental monitoring centers that oversee protected areas and biodiversity. "
            "The Cerrado is Brazil's second-largest biome, known for its rich biodiversity and significant role in water resources. "
            "INPE (Instituto Nacional de Pesquisas Espaciais) is Brazil's National Institute for Space Research, responsible for monitoring deforestation and environmental changes using satellite data. "
            "The teacher provides explanations, while the students ask questions and engage in the discussion."
        )

        # Whisper transcription with tuned parameters
        result = model.transcribe(
            audio=input_wav,
            language=self.language,
            task='transcribe',
            verbose=True,
            temperature=0.0,
            beam_size=5,
            prompt=prompt
        )
        segments = result.get('segments', [])

        # PyAnnote diarization
        logger.info("Executando diarização com pyannote")

        device = self.get_device()

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1").to(device)

        diarization = pipeline(input_wav, min_speakers=2, max_speakers=5)
        diar_segments = list(diarization.itertracks(yield_label=True))

        # Align speaker labels to Whisper segments
        final_output = []
        for seg in segments:
            start, end, text = seg['start'], seg['end'], seg['text']
            speaker_label = "Unknown"

            # Find overlapping diarization segments
            overlaps = [
                (turn, speaker)
                for (turn, _, speaker) in diar_segments
                if not (turn.end <= start or turn.start >= end)
            ]

            if overlaps:
                # Choose speaker with longest overlapping duration
                speaker_counts = {}
                for (turn, speaker) in overlaps:
                    overlap_start = max(turn.start, start)
                    overlap_end = min(turn.end, end)
                    duration = overlap_end - overlap_start
                    speaker_counts[speaker] = speaker_counts.get(speaker, 0) + duration

                speaker_label = max(speaker_counts, key=speaker_counts.get)

            final_output.append({
                "speaker": speaker_label,
                "start": start,
                "end": end,
                "text": text.strip()
            })

        # Save to JSON
        out_name = f"{Path(input_wav).stem}_transcription.json"
        out_path = os.path.join(self.results_dir, out_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        logger.info("Transcrição com falantes salva em: %s", out_path)
        return out_path

    def executar_pipeline(self) -> None:
        logger.info("=== Iniciando pipeline completo ===")
        found = self.find_media_file()
        if not found:
            return
        media_file, is_video = found
        base_name = Path(media_file).stem
        logger.info("Arquivo para processar: %s (video=%s)", media_file, is_video)

        # 1) Converter vídeo/áudio original para WAV mono 16kHz
        ext = Path(media_file).suffix.lower()
        wav = self.convert_to_wav(media_file) if is_video or ext != '.wav' else media_file

        # 2) Redução de ruído
        denoised = self.reduce_stationary_noise(wav)

        # 3) Separação de vocais
        vocals = self.extract_vocals(denoised)

        # 4) Normalização básica de volume
        norm = self.normalize_audio(vocals)

        # 5) Normalização de loudness (LUFS) sem clipping
        norm_l = self.normalize_loudness(norm)


        # 6) Equalização de locutores
        try:
            voz_eq = self.equalize_speakers(norm_l)
        except Exception as e:
            logger.error("Erro na equalização de locutores: %s", e)
            raise RuntimeError("Erro na equalização de locutores") from e

        # 7) Remoção de silêncio/respirações
        voz_clean = self.remove_silence(voz_eq)

        # 8) Filtragem VAD
        voz_vad = self.vad_filter(voz_clean, base_name)

        # 9) Transcrição
        if voz_eq:
            try:
                self.transcrever_audio(voz_vad)
            except Exception as e:
                logger.error("Erro na transcrição: %s", e)

        logger.info("Pipeline concluído com sucesso")


def main():
    parser = argparse.ArgumentParser(
        description="Processa arquivos de mídia e transcreve áudio usando OpenAI Whisper"
    )
    parser.add_argument(
        '--media-dir', default='./files', help='Diretório contendo arquivos de mídia'
    )
    parser.add_argument(
        '--model', default='large-v3-turbo', help='Modelo de transcrição OpenAI'
    )
    parser.add_argument(
        '--language', default='pt', help='Código de idioma para transcrição'
    )
    args = parser.parse_args()

    load_dotenv()

    key = os.getenv('OPENAI_API_KEY')
    if not key:
        logger.error('API key do OpenAI não encontrada. Defina OPENAI_API_KEY.')
    openai.api_key = key

    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise RuntimeError('Token do Hugging Face não encontrado. Defina HF_TOKEN.')
    HfFolder.save_token(hf_token)

    processor = ProcessadorAudio(
        media_dir=args.media_dir,
        model=args.model,
        language=args.language
    )
    processor.executar_pipeline()


if __name__ == '__main__':
    main()
