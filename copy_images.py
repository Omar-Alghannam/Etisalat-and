import os
import shutil

src_dir = r"C:\Users\Omar\.gemini\antigravity-ide\brain\3373c860-4807-483d-aa43-9116ae27ff19\.user_uploaded"
dest_dir = r"c:\E&_Agentic\images"

os.makedirs(dest_dir, exist_ok=True)

mapping = {
    "media_1787939415544.png": "emerald.png",
    "media_1787939426653.png": "hekaya_internet.png",
    "media_1787939431258.png": "hekaya_mixat.png",
    "media_1787939458018.png": "aqwa_card.png",
    "media_1787939513813.png": "dataline.png"
}

for src_name, dest_name in mapping.items():
    src_path = os.path.join(src_dir, src_name)
    dest_path = os.path.join(dest_dir, dest_name)
    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
        print(f"Copied {src_name} to {dest_name}")
    else:
        print(f"File not found: {src_path}")
