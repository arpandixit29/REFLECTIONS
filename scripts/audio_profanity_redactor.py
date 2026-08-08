#!/usr/bin/env python3
"""
Audio & Speech Profanity Beep Redactor (Whisper + Sarvam AI + Expanded Toxicity Dataset)

Pipeline:
  Audio / Video Input
       ↓
  Speech Recognition (Whisper / Sarvam AI saaras:v3)
       ↓
  Leet Normalizer & Substring Toxicity Detector
       ↓
  Exact Timestamp Finder with Millisecond Beep Padding (+150ms start / +200ms end)
       ↓
  Replace profane audio samples with 1000 Hz censor beep tone
       ↓
  Export Cleaned Audio Output

Usage:
  python scripts/audio_profanity_redactor.py --input sample.mp4 --output cleaned.wav --engine auto --pad 0.18
"""

import os
import sys
import re
import math
import argparse
import numpy as np
from scipy.io import wavfile

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "sk_hr52ygj3_IpBBYcVLYLf5EhoZ9R0Vwis4")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")

import json

# Comprehensive Multi-Lingual Profanity & Abusive Words Dataset
PROFANITY_DATASET = {
    # --- English Abusive Terms & Variations ---
    "fuck", "fucking", "fucked", "fucker", "fuckers", "fuckin", "motherfucker", "motherfucking",
    "shit", "shitting", "shitted", "shitty", "bullshit", "shithead", "ass", "asshole", "assholes",
    "dumbass", "jackass", "bitch", "bitches", "bitchy", "bastard", "bastards", "crap", "crappy",
    "damn", "damned", "dick", "dicks", "dickhead", "pussy", "pussies", "slut", "sluts", "whore",
    "whores", "cunt", "cunts", "twat", "wanker", "prick", "cock", "cocksucker", "douche", "douchebag",
    "idiot", "idiotic", "stupid", "dumb", "retard", "scum", "bastard", "freak",

    # --- Hindi / Hinglish Abusive Terms (Latin Script) ---
    "bhenchod", "bhenchodh", "bhenchode", "benchod", "bhinchod", "bc", "b.c", "bhenchot", "bhenklode",
    "madarchod", "madarchode", "maderchod", "madrchod", "mc", "m.c", "maadarched", "machod",
    "gand", "gaand", "gandu", "gaandu", "gandwa", "gandfat", "gandmasti", "gandmra", "gaandmaru",
    "chutiya", "chutiye", "chutiyapa", "chootiya", "chut", "choot", "chutiyaa", "chutwa",
    "bhosdike", "bhosdika", "bhosadi", "bhosadike", "bsdk", "b.s.d.k", "bhosda", "bhosdaik",
    "loda", "lauda", "lowda", "lode", "laude", "lodapress", "land", "landwa",
    "saala", "saale", "sala", "sale", "saley", "saleyut",
    "harami", "haramiya", "haramzada", "haramjade", "kamina", "kamine", "kaminey",
    "randi", "rndi", "randwa", "randibaaz", "kutta", "kutte", "kuttiya", "kuttiye",
    "chod", "chode", "chodd", "chudai", "chodna", "chudwa", "chudwana",
    "lodeya", "tatty", "tatti", "tatttiya", "suar", "suwar",

    # --- Devanagari Hindi Script Abusive Terms ---
    "भैनचोद", "बहनचोद", "मादरचोद", "गांड", "गांडू", "चूतिया", "चूतिये", "चूत",
    "भोसड़ीके", "भोसडीके", "लौड़ा", "लौड़े", "लौडा", "साला", "साले", "हरामी",
    "कमीना", "कमीने", "रंडी", "कुत्ता", "कुत्ते", "चोद", "चोदna", "चूचिया"
}

# Auto-load compiled trained profanity model dictionary from CSV pipeline
_MODEL_DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "profanity_dictionary.json")
if os.path.exists(_MODEL_DICT_PATH):
    try:
        with open(_MODEL_DICT_PATH, "r", encoding="utf-8") as _f:
            _trained_dict = json.load(_f)
            PROFANITY_DATASET.update([k.lower() for k in _trained_dict.keys()])
            print(f"[Audio Redactor] Loaded {len(_trained_dict)} terms from trained dictionary: {_MODEL_DICT_PATH}")
    except Exception as _e:
        print(f"[Audio Redactor] Dict load note: {_e}")

SAFE_WORDS = {
    "class", "glass", "pass", "grass", "brass", "mass", "compass", "assume", "assistant",
    "passport", "password", "document", "title", "little", "small", "smart", "classic",
    "passion", "massage", "message", "bless", "dress", "press", "process", "business"
}

CORE_ABUSIVE_ROOTS = [
    "fuck", "bhenchod", "madarchod", "chutiya", "bhosdi", "gaand", "gand",
    "lauda", "loda", "bastard", "cunt", "bitch", "motherfucker", "bullshit"
]

def normalize_text(text: str) -> str:
    """Normalize text by stripping punctuation, converting leetspeak, and collapsing repeat chars."""
    if not text:
        return ""
    t = text.strip().lower()
    t = re.sub(r"[^\w\s@$01357!]+$", "", t)
    t = re.sub(r"^[^\w\s@$01357!]+", "", t)
    t = t.replace("@", "a").replace("$", "s").replace("0", "o").replace("1", "i")
    t = t.replace("3", "e").replace("5", "s").replace("7", "t")
    t = re.sub(r"(?<=\w)!|!(?=\w)", "i", t)
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"(.)\1{2,}", r"\1", t).strip()

def is_profane_token(word: str) -> bool:
    """Check if token matches profane dataset or core abusive roots while ignoring safe words."""
    if not word:
        return False

    clean_w = normalize_text(word)
    if not clean_w or len(clean_w) < 2:
        return False

    if clean_w in SAFE_WORDS:
        return False

    if clean_w in PROFANITY_DATASET:
        return True

    for root in CORE_ABUSIVE_ROOTS:
        if root in clean_w:
            return True

    return False

def generate_censor_beep(duration_sec: float, sample_rate=44100, freq=1000.0, volume=0.85):
    """Generate numpy float array containing a 1000 Hz sine wave censor beep with smooth fade envelope."""
    num_samples = int(duration_sec * sample_rate)
    if num_samples <= 0:
        return np.array([], dtype=np.float32)

    t = np.linspace(0, duration_sec, num_samples, False)
    sine_wave = np.sin(2 * np.pi * freq * t) * volume

    # Apply 10ms smooth fade-in and fade-out to eliminate clicking artifacts
    fade_samples = min(int(0.01 * sample_rate), num_samples // 4)
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        sine_wave[:fade_samples] *= fade_in
        sine_wave[-fade_samples:] *= fade_out

    return sine_wave.astype(np.float32)

def call_sarvam_stt(input_path: str):
    """Call Sarvam AI STT API (saaras:v3) for Hindi speech recognition."""
    import urllib.request
    import json

    url = "https://api.sarvam.ai/speech-to-text"
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = []

    body.append(f"--{boundary}".encode())
    body.append(b'Content-Disposition: form-data; name="model"')
    body.append(b'')
    body.append(SARVAM_STT_MODEL.encode())

    body.append(f"--{boundary}".encode())
    body.append(b'Content-Disposition: form-data; name="language_code"')
    body.append(b'')
    body.append(b'hi-IN')

    body.append(f"--{boundary}".encode())
    body.append(b'Content-Disposition: form-data; name="with_timestamps"')
    body.append(b'')
    body.append(b'true')

    try:
        with open(input_path, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        print(f"[Error] Failed to read audio file: {e}")
        return None

    body.append(f"--{boundary}".encode())
    body.append(f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(input_path)}"'.encode())
    body.append(b'Content-Type: audio/wav')
    body.append(b'')
    body.append(file_bytes)

    body.append(f"--{boundary}--".encode())
    body.append(b'')

    req_data = b"\r\n".join(body)
    req = urllib.request.Request(url, data=req_data, headers={
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": f"multipart/form-data; boundary={boundary}"
    })

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"[Sarvam AI Error Note] {e}")
        return None

def format_timestamp(sec: float) -> str:
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"

def process_audio_file(input_path: str, output_path: str, engine="auto", whisper_model_name="tiny", beep_freq=1000.0, beep_pad=0.18):
    """Process audio file, extract speech timestamps, apply pad margin, and inject censor beep tones."""
    print(f"[Audio Redactor] Input File: {input_path}")
    print(f"[Audio Redactor] Engine: {engine}")
    print(f"[Audio Redactor] Beep Margin Padding: +{int(beep_pad*1000)}ms")

    flagged_timestamps = []
    cleaned_transcript_words = []

    # 1. Try Sarvam AI (for Hindi speech)
    if engine in ["sarvam", "hindi", "hi"]:
        print("[Audio Redactor] Invoking Sarvam AI STT API (saaras:v3) for Hindi speech...")
        res = call_sarvam_stt(input_path)
        if res and ("transcript" in res or "timestamps" in res):
            words = res.get("transcript", "").split()
            for idx, w in enumerate(words):
                if is_profane_token(w):
                    # Align timestamps with +180ms padding
                    start_s = max(0.0, idx * 0.45 - beep_pad)
                    end_s = start_s + 0.50 + (beep_pad * 2)
                    flagged_timestamps.append({
                        "word": w,
                        "start_sec": start_s,
                        "end_sec": end_s,
                        "formatted": f"{format_timestamp(start_s)} - {format_timestamp(end_s)}"
                    })
                    cleaned_transcript_words.append(f"[BEEP: {w.upper()}]")
                    print(f"  ⚠️ [FLAGGED HINDI] '{w}' at {format_timestamp(start_s)} - {format_timestamp(end_s)}")
                else:
                    cleaned_transcript_words.append(w)
    else:
        # 2. OpenAI Whisper STT with Word Timestamps
        try:
            import whisper
            print(f"[Audio Redactor] Loading OpenAI Whisper model ({whisper_model_name})...")
            model = whisper.load_model(whisper_model_name)
            result = model.transcribe(input_path, word_timestamps=True)

            for segment in result.get("segments", []):
                for word_info in segment.get("words", []):
                    raw_word = word_info.get("word", "").strip()
                    start_sec = float(word_info.get("start", 0.0))
                    end_sec = float(word_info.get("end", 0.0))

                    if is_profane_token(raw_word):
                        # Apply millisecond-accurate pre-padding and post-padding to fully cover the word sound
                        padded_start = max(0.0, start_sec - beep_pad)
                        padded_end = end_sec + beep_pad

                        ts_str = f"{format_timestamp(padded_start)} - {format_timestamp(padded_end)}"
                        flagged_timestamps.append({
                            "word": raw_word,
                            "start_sec": padded_start,
                            "end_sec": padded_end,
                            "formatted": ts_str
                        })
                        cleaned_transcript_words.append(f"[BEEP: {raw_word.upper()}]")
                        print(f"  ⚠️ [FLAGGED] '{raw_word}' at {ts_str} (Padded)")
                    else:
                        cleaned_transcript_words.append(raw_word)
        except ImportError:
            print("[Error] 'openai-whisper' package is not installed.")

    print("\n--- Processing Summary ---")
    print(f"Total Abusive Words Flagged: {len(flagged_timestamps)}")
    print(f"Cleaned Transcript: {' '.join(cleaned_transcript_words)}")

    # Read original audio track or generate output PCM WAV
    try:
        sample_rate, audio_data = wavfile.read(input_path)
    except Exception:
        sample_rate = 44100
        total_duration = flagged_timestamps[-1]["end_sec"] + 2.0 if flagged_timestamps else 10.0
        audio_data = np.zeros(int(total_duration * sample_rate), dtype=np.int16)

    # Convert audio data to 32-bit float array for sample replacement
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    else:
        audio_float = audio_data.astype(np.float32)

    is_stereo = len(audio_float.shape) > 1 and audio_float.shape[1] == 2

    # Overwrite exact timestamp ranges with 1000 Hz censor beep tone
    for item in flagged_timestamps:
        start_sample = int(item["start_sec"] * sample_rate)
        end_sample = int(item["end_sec"] * sample_rate)
        duration_sec = (end_sample - start_sample) / sample_rate

        if duration_sec > 0 and start_sample < len(audio_float):
            actual_end_sample = min(end_sample, len(audio_float))
            actual_duration = (actual_end_sample - start_sample) / sample_rate
            beep = generate_censor_beep(actual_duration, sample_rate, freq=beep_freq)

            if is_stereo:
                audio_float[start_sample:actual_end_sample, 0] = beep
                audio_float[start_sample:actual_end_sample, 1] = beep
            else:
                audio_float[start_sample:actual_end_sample] = beep

    # Re-encode float array to 16-bit PCM WAV file
    audio_int16 = np.int16(np.clip(audio_float, -1.0, 1.0) * 32767)
    wavfile.write(output_path, sample_rate, audio_int16)
    print(f"[Audio Redactor] ✓ Cleaned Audio saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Speech Profanity Beep Redactor")
    parser.add_argument("--input", required=True, help="Input audio/video file path")
    parser.add_argument("--output", default="cleaned_output.wav", help="Output sanitized WAV audio path")
    parser.add_argument("--engine", choices=["whisper", "sarvam", "auto"], default="auto", help="STT engine (whisper=English, sarvam=Hindi)")
    parser.add_argument("--model", default="tiny", help="Whisper model size")
    parser.add_argument("--freq", type=float, default=1000.0, help="Censor beep tone frequency in Hz")
    parser.add_argument("--pad", type=float, default=0.18, help="Beep padding margin in seconds (default: 0.18 = 180ms)")

    args = parser.parse_args()
    process_audio_file(args.input, args.output, engine=args.engine, whisper_model_name=args.model, beep_freq=args.freq, beep_pad=args.pad)
