import os
import subprocess
import glob

original_dir = r"d:\machine_learning\papers\original_en"
output_dir = r"d:\machine_learning\papers\translated_zh"
os.makedirs(output_dir, exist_ok=True)

pdf_files = glob.glob(os.path.join(original_dir, "*.pdf"))

for pdf_path in pdf_files:
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    dual_pdf = os.path.join(output_dir, f"{base_name}-dual.pdf")
    mono_pdf = os.path.join(output_dir, f"{base_name}-mono.pdf")
    
    # Check if already translated
    if os.path.exists(dual_pdf) and os.path.getsize(dual_pdf) > 1000000 and "UNet++" not in base_name:
        print(f"Already translated: {base_name}")
        continue
        
    print(f"\n>>> Translating core pages (1-5) of {base_name}...")
    # Translate core pages 1-5 (Abstract, Method, Network Architecture, Results)
    cmd = [
        "pdf2zh",
        pdf_path,
        "-p", "1-5",
        "-o", output_dir,
        "--service", "bing"
    ]
    try:
        ret = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Done translating: {base_name}")
    except Exception as e:
        print(f"Error translating {base_name}: {e}")
