import urllib.request
import os
import json

target_dir = r"c:\Users\Arpan\Desktop\reflections\web\model"
os.makedirs(target_dir, exist_ok=True)

base_url = "https://raw.githubusercontent.com/infinitered/nsfwjs/master/models/mobilenet_v2/"

print("Downloading model.json...")
req = urllib.request.Request(base_url + "model.json", headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    content = response.read()
    with open(os.path.join(target_dir, "model.json"), "wb") as f:
        f.write(content)

print("Reading model.json to find weight shard filenames...")
model_data = json.loads(content.decode('utf-8'))
weights_manifest = model_data.get("weightsManifest", [])

shard_files = []
for manifest in weights_manifest:
    for path in manifest.get("paths", []):
        shard_files.append(path)

print("Found shard files:", shard_files)

for shard in shard_files:
    print(f"Downloading shard: {shard}...")
    s_url = base_url + shard
    s_req = urllib.request.Request(s_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(s_req) as response:
        s_content = response.read()
        with open(os.path.join(target_dir, shard), "wb") as f:
            f.write(s_content)

print("SUCCESSFULLY DOWNLOADED ALL MODEL FILES LOCALLY!")
