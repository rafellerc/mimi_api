import argparse
import csv
import glob
import os
import time
from typing import List

import requests


API_URL = "http://0.0.0.0:8080/process-audio"


def post_audio(audio_path: str) -> dict:
    """
    POST a single audio file to the Mimi API and return the JSON response.
    """
    with open(audio_path, "rb") as f:
        files = {"audio": (os.path.basename(audio_path), f)}
        response = requests.post(API_URL, files=files, timeout=900)

    response.raise_for_status()
    return response.json()


def warmup(audios: List[str], n_warmup: int) -> None:
    """
    Run warmup requests (responses are ignored).
    """
    if n_warmup <= 0:
        return

    print(f"Running warmup on {n_warmup} audio(s)...")

    for audio_path in audios[:n_warmup]:
        try:
            _ = post_audio(audio_path)
        except Exception as e:
            print(f"Warmup failed for {audio_path}: {e}")

    print("Warmup completed.\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Mimi audio tokenizer API")
    parser.add_argument(
        "--sample-dir",
        default="./sample",
        help="Directory containing .flac samples",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run identifier (e.g. cpu_auto, cuda_t4, etc.)",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=1,
        help="Number of warmup requests",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.csv",
        help="Output CSV file",
    )

    args = parser.parse_args()

    audio_paths = sorted(glob.glob(os.path.join(args.sample_dir, "*.flac")))

    if not audio_paths:
        raise RuntimeError(f"No .flac files found in {args.sample_dir}")

    print(f"Found {len(audio_paths)} audio files.")

    # -------------------------
    # Warmup phase
    # -------------------------
    warmup(audio_paths, args.n_warmup)

    # -------------------------
    # Benchmark phase
    # -------------------------
    results = []

    print("Running benchmark...\n")

    for audio_path in audio_paths:
        print(f"Processing {os.path.basename(audio_path)}")

        try:
            response = post_audio(audio_path)

            results.append(
                {
                    "run_name": args.run_name,
                    "audio_file": os.path.basename(audio_path),
                    "duration_seconds": response.get("duration_seconds"),
                    "latency_ms": response.get("latency_ms"),
                    "device": response.get("device"),
                    "num_codebooks": response.get("num_codebooks"),
                }
            )

        except Exception as e:
            print(f"Failed for {audio_path}: {e}")

        # Optional small delay to avoid overwhelming the server
        time.sleep(0.05)

    # -------------------------
    # Write CSV
    # -------------------------
    fieldnames = [
        "run_name",
        "audio_file",
        "duration_seconds",
        "latency_ms",
        "device",
        "num_codebooks",
    ]

    write_header = not os.path.exists(args.output)

    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerows(results)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
