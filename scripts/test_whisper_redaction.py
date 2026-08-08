import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import static_ffmpeg
static_ffmpeg.add_paths()
import whisper
import json
import re

# Load 1,487-term dataset
dict_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "profanity_dictionary.json")
with open(dict_path, "r", encoding="utf-8") as f:
    PROFANITY_DICT = set(json.load(f))

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

def is_profane(word: str) -> bool:
    clean = normalize_text(word)
    if not clean or len(clean) < 2:
        return False
    if clean in SAFE_WORDS:
        return False
    if clean in PROFANITY_DICT:
        return True
    for root in CORE_ABUSIVE_ROOTS:
        if root in clean:
            return True
    return False

print(f"[Test] Profanity Dataset loaded: {len(PROFANITY_DICT)} terms")
print("--- Testing Abusive Terms ---")
print(f"  'fuck' profane?        {is_profane('fuck')}")
print(f"  'fucking!' profane?    {is_profane('fucking!')}")
print(f"  'bhenchod' profane?    {is_profane('bhenchod')}")
print(f"  'madarchod' profane?   {is_profane('madarchod')}")
print(f"  'chutiya' profane?     {is_profane('chutiya')}")
print(f"  'bsdk' profane?        {is_profane('bsdk')}")
print(f"  'gaand' profane?       {is_profane('gaand')}")
print(f"  'bhosdike' profane?    {is_profane('bhosdike')}")

print("\n--- Testing Safe Non-Profane Words (Should all be False) ---")
print(f"  'class' profane?       {is_profane('class')}")
print(f"  'glass' profane?       {is_profane('glass')}")
print(f"  'pass' profane?        {is_profane('pass')}")
print(f"  'bhai' profane?        {is_profane('bhai')}")
print(f"  'aadmi' profane?       {is_profane('aadmi')}")
print(f"  'document' profane?    {is_profane('document')}")

