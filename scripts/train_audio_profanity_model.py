#!/usr/bin/env python3
"""
Reflections Audio Profanity Model Trainer
Supports multi-CSV training from uploaded datasets.

Usage:
  python scripts/train_audio_profanity_model.py
  python scripts/train_audio_profanity_model.py --csv data/my_custom.csv
"""

import os
import sys
import csv
import json
import re
import argparse

# ─── Stop-words to filter out (non-profane filler words in phrases) ──────────
STOPWORDS = {
    "a", "an", "the", "i", "you", "u", "he", "she", "they", "we", "me", "him",
    "her", "it", "is", "was", "are", "were", "be", "been", "has", "have", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "so", "very", "that", "this", "and", "or", "but", "not", "no", "yes", "ok",
    "okay", "lol", "bro", "dude", "man", "like", "ya", "yaar", "yaaaar", "yar",
    "ho", "hai", "hain", "haai", "haain", "haee", "haeen", "tha", "thaa", "thi",
    "hey", "such", "what", "how", "why", "when", "just", "tu", "tum", "aap",
    "wo", "woh", "yeh", "aur", "ke", "ki", "ka", "se", "pe", "par", "bhi", "hi",
    "na", "nahi", "kya", "hoga", "aadmi", "aaaadmi", "aadmee", "admi", "insaan",
    "insaaaan", "insan", "insani", "eensaan", "bhai", "ladka", "ladki", "aurat",
    "bachcha", "human", "artist", "attention", "brain", "control", "dekh",
    "face", "faced", "get", "harkat", "internet", "keyboard", "lost", "of",
    "off", "oye", "pathe", "piece", "seeker", "soch", "son", "sun", "two",
    "up", "wala", "warrior", "work", "aalsi", "aalsee", "aaaalsi", "alsi",
    "class", "glass", "pass", "grass", "brass", "mass", "document", "title"
}

def normalize_token(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    # Strip trailing punctuation first so exclamation points at the end of words ('fuck!') don't become 'fucki'
    t = re.sub(r"[^\w\s@$01357!]+$", "", t)
    t = re.sub(r"^[^\w\s@$01357!]+", "", t)
    # Substitute leetspeak symbols
    t = t.replace("@", "a").replace("$", "s").replace("0", "o").replace("1", "i")
    t = t.replace("3", "e").replace("5", "s").replace("7", "t")
    # Replace ! with i only when inside words (e.g. sh!t, f!ck)
    t = re.sub(r"(?<=\w)!|!(?=\w)", "i", t)
    # Strip all non-alphanumeric except spaces
    t = re.sub(r"[^\w\s]", "", t)
    # Collapse repeated consecutive characters (fuuuuck -> fuck, bhenchhhhhod -> bhenchod)
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    return t.strip()

def extract_root_words(phrase: str):
    """
    Extract meaningful root words from a phrase like 'you are such a fuck bro'
    by stripping stopwords and returning unique non-empty tokens.
    """
    tokens = phrase.strip().split()
    roots = set()
    for tok in tokens:
        norm = normalize_token(tok)
        if norm and norm not in STOPWORDS and len(norm) >= 3:
            roots.add(norm)
    return roots

def generate_variations(word: str):
    """Generate common phonetic/spelling variations for a root word."""
    clean = normalize_token(word)
    variations = {clean}
    # Phonetic swaps
    variations.add(clean.replace("ph", "f"))
    variations.add(clean.replace("ck", "k"))
    variations.add(clean.replace("ck", "c"))
    variations.add(clean.replace("ch", "chh"))
    variations.add(clean.replace("aa", "a"))
    variations.add(clean.replace("ee", "i"))
    variations.add(clean.replace("a", "aa"))
    # Suffix drops
    if clean.endswith("ing"):
        variations.add(clean[:-3])
    if clean.endswith("ed"):
        variations.add(clean[:-2])
    if clean.endswith("er"):
        variations.add(clean[:-2])
    if clean.endswith("s") and len(clean) > 4:
        variations.add(clean[:-1])
    return {v for v in variations if v and len(v) >= 3 and v not in STOPWORDS}

def train_from_csvs(csv_paths: list, output_json: str):
    profanity_map = {}
    total_rows = 0
    total_files = 0

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"[Skip] File not found: {csv_path}")
            continue

        print(f"[Trainer] Processing: {csv_path}")
        total_files += 1

        with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            # Auto-detect column positions
            text_col, label_col, lang_col = 0, 1, 2
            if header:
                for idx, col in enumerate(header):
                    cn = col.strip().lower()
                    if cn in ("text", "word", "term", "phrase"):
                        text_col = idx
                    elif cn in ("label", "category", "severity"):
                        label_col = idx
                    elif cn in ("language", "lang"):
                        lang_col = idx

            for row in reader:
                if not row or len(row) <= text_col:
                    continue

                raw_text = row[text_col].strip()
                label_raw = row[label_col].strip() if len(row) > label_col else "1"
                lang = row[lang_col].strip() if len(row) > lang_col else "en"

                # Only include labeled abusive rows (label == 1 or "profanity"/"explicit"/"high"/"critical")
                label_lower = label_raw.lower()
                is_abusive = (
                    label_raw == "1" or
                    label_lower in ("profanity", "explicit", "high", "critical", "abusive", "hate", "slur", "insult", "medium")
                )
                if not is_abusive:
                    continue

                # Extract root words from the phrase
                root_words = extract_root_words(raw_text)
                total_rows += 1

                for root in root_words:
                    for var in generate_variations(root):
                        if var not in profanity_map:
                            profanity_map[var] = {
                                "original": root,
                                "language": lang,
                                "source_phrase": raw_text[:80]
                            }

    # Save compiled model checkpoint to main models directory and web model directory
    output_dir = os.path.dirname(output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(profanity_map, f, indent=2, ensure_ascii=False)

    # Export copy to web/model/ for frontend web app
    web_model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "model")
    if os.path.exists(web_model_dir):
        web_json_path = os.path.join(web_model_dir, "profanity_dictionary.json")
        with open(web_json_path, "w", encoding="utf-8") as f:
            json.dump(profanity_map, f, indent=2, ensure_ascii=False)
        print(f"[OK] Web Model exported to: {web_json_path}")

    print("\n--- Training Complete ---")
    print(f"CSV files processed       : {total_files}")
    print(f"Abusive phrase rows used  : {total_rows}")
    print(f"Compiled vocabulary terms : {len(profanity_map)}")
    print(f"[OK] Model exported to    : {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Profanity Model from Uploaded CSV Datasets")
    parser.add_argument(
        "--csv",
        nargs="+",
        default=[
            "data/english_abusive_words_extended (1).csv",
            "data/roman_hindi_abusive_dataset_30000.csv",
            "data/profanity_dataset.csv"
        ],
        help="One or more CSV dataset files to train from"
    )
    parser.add_argument("--output", default="models/profanity_dictionary.json",
                        help="Output JSON model checkpoint path")

    args = parser.parse_args()
    train_from_csvs(args.csv, args.output)
