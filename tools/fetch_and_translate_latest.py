import os
import urllib.request
import subprocess

pdf_url = "https://arxiv.org/pdf/2402.02491.pdf"
save_path = r"d:\machine_learning\papers\original_en\2025_VM-UNet_Mamba.pdf"
output_dir = r"d:\machine_learning\papers\translated_zh"

print("Downloading VM-UNet (Visual Mamba)...")
headers = {'User-Agent': 'Mozilla/5.0'}
req = urllib.request.Request(pdf_url, headers=headers)
with urllib.request.urlopen(req, timeout=30) as resp, open(save_path, 'wb') as f:
    f.write(resp.read())
print(f"Downloaded 2025_VM-UNet_Mamba.pdf ({os.path.getsize(save_path)} bytes)")

print("Translating core pages 1-4 with pdf2zh...")
cmd = [
    "pdf2zh",
    save_path,
    "-p", "1-4",
    "-o", output_dir,
    "--service", "bing"
]
ret = subprocess.run(cmd, capture_output=True, text=True, check=True)
print("Translation complete!")
