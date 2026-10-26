import os
import urllib.request

papers = [
    {
        "filename": "2024_NatComm_MedSAM.pdf",
        "title": "Segment Anything in Medical Images (MedSAM)",
        "journal": "Nature Communications (2024, IF: 14.7)",
        "url": "https://arxiv.org/pdf/2304.12306.pdf",
        "alt_url": "https://www.nature.com/articles/s41467-024-44824-z.pdf"
    },
    {
        "filename": "2020_IEEE_TMI_UNet++.pdf",
        "title": "UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation",
        "journal": "IEEE Transactions on Medical Imaging (IEEE TMI, IF: 8.9)",
        "url": "https://arxiv.org/pdf/1807.10165.pdf"
    },
    {
        "filename": "2018_Attention_UNet.pdf",
        "title": "Attention U-Net: Learning Where to Look for the Pancreas",
        "journal": "MIDL / Medical Image Analysis (MedIA, IF: 10.9)",
        "url": "https://arxiv.org/pdf/1804.03999.pdf"
    },
    {
        "filename": "2021_TransUNet.pdf",
        "title": "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation",
        "journal": "arXiv / Medical AI Landmark (Cited 4000+)",
        "url": "https://arxiv.org/pdf/2102.04306.pdf"
    }
]

dest_dir = r"d:\machine_learning\papers\original_en"
os.makedirs(dest_dir, exist_ok=True)

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for p in papers:
    save_path = os.path.join(dest_dir, p["filename"])
    if os.path.exists(save_path) and os.path.getsize(save_path) > 50000:
        print(f"Already exists: {p['filename']} ({os.path.getsize(save_path)} bytes)")
        continue
    print(f"Downloading {p['title']} from {p['url']}...")
    try:
        req = urllib.request.Request(p["url"], headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp, open(save_path, 'wb') as out_f:
            out_f.write(resp.read())
        print(f"Successfully downloaded {p['filename']} ({os.path.getsize(save_path)} bytes)")
    except Exception as e:
        print(f"Failed to download {p['filename']}: {e}")
