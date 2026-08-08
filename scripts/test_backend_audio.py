import os
import numpy as np
from scipy.io import wavfile
import urllib.request
import json


# Create a sample 3-second WAV file
sample_rate = 16000
duration = 3.0
t = np.linspace(0, duration, int(sample_rate * duration), False)
# Audio signal (sine wave)
audio = (np.sin(2 * np.pi * 440 * t) * 16384).astype(np.int16)

wav_path = "test_sample.wav"
wavfile.write(wav_path, sample_rate, audio)

print(f"[Test] Generated {wav_path}")

# Send to backend
url = "http://127.0.0.1:8000/v1/transcribe_and_beep"
boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = []

body.append(f"--{boundary}".encode())
body.append(b'Content-Disposition: form-data; name="file"; filename="test_sample.wav"')
body.append(b'Content-Type: audio/wav')
body.append(b'')
with open(wav_path, "rb") as f:
    body.append(f.read())

body.append(f"--{boundary}--".encode())
body.append(b'')

req_data = b"\r\n".join(body)
req = urllib.request.Request(url, data=req_data, headers={
    "Content-Type": f"multipart/form-data; boundary={boundary}"
})

try:
    with urllib.request.urlopen(req, timeout=30) as response:
        res = json.loads(response.read().decode("utf-8"))
        print("[Test] Endpoint response SUCCESS!")
        print("  Status:", res.get("status"))
        print("  Profanities detected:", res.get("profanities_detected"))
        print("  Message:", res.get("message"))
except Exception as e:
    print("[Test Error]:", e)

if os.path.exists(wav_path):
    os.remove(wav_path)
