
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "results"
PRETRAINED_MODELS_DIR = BASE_DIR / "pretrained_models"

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
PRETRAINED_MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATHS = {
    "vad": str(PRETRAINED_MODELS_DIR / "FireRedVAD" / "VAD"),
    "asr": str(PRETRAINED_MODELS_DIR / "FireRedASR2-AED"),
    "punc": str(PRETRAINED_MODELS_DIR / "FireRedPunc"),
}

ASR_CONFIG = {
    "use_gpu": True,
    "use_half": False,
    "beam_size": 3,
    "nbest": 1,
    "decode_max_len": 0,
    "softmax_smoothing": 1.25,
    "aed_length_penalty": 0.6,
    "eos_penalty": 1.0,
    "return_timestamp": True,
}

VAD_CONFIG = {
    "use_gpu": False,
    "smooth_window_size": 5,
    "speech_threshold": 0.4,
    "min_speech_frame": 20,
    "max_speech_frame": 2000,
    "min_silence_frame": 20,
    "merge_silence_frame": 0,
    "extend_speech_frame": 0,
    "chunk_max_frame": 30000,
}

PUNC_CONFIG = {
    "use_gpu": True,
}

BATCH_CONFIG = {
    "asr_batch_size": 8,
}
