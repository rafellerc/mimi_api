import os
import time
from flask import Flask, request, jsonify, abort
import torch
import librosa
from datasets import Audio
from transformers import MimiModel, AutoFeatureExtractor
import torch
from utils import get_device


# -------------------------
# Configuration
# -------------------------
DEVICE = get_device()
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20 MB upload limit
MAX_AUDIO_SECONDS = 60                 # max allowed duration
MODEL_ID = "kyutai/mimi"

torch.set_grad_enabled(False)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Initialize the mimi processor and model from Hugging Face
model = MimiModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)


@app.route("/process-audio", methods=["POST"])
def process_audio():
    if "audio" not in request.files:
        abort(400, description="Missing audio file")

    start_time = time.perf_counter()

    audio_file = request.files["audio"]

    # Decode + resample + downmix
    waveform, sr = librosa.load(
        audio_file,
        sr=feature_extractor.sampling_rate,  # 24 kHz
        mono=True
    )

    duration_sec = len(waveform) / sr
    if duration_sec > MAX_AUDIO_SECONDS:
        abort(
            400,
            description=f"Audio too long ({duration_sec:.2f}s). "
                        f"Max allowed is {MAX_AUDIO_SECONDS}s."
        )

    # Prepare input
    inputs = feature_extractor(
        raw_audio=waveform,
        sampling_rate=sr,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = outputs.audio_codes.tolist()

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return jsonify({
        "audio_codes": tokens,
        "sample_rate": sr,
        "duration_seconds": round(duration_sec, 3),
        "num_codebooks": len(tokens[0]) if tokens else None,
        "model_id": MODEL_ID,
        "axes": ["codebook", "time"],
        "device": str(DEVICE),
        "latency_ms": round(elapsed_ms, 2)
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "mimi-audio-tokenizer",
        "device": str(DEVICE)
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
