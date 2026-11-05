# 🧾 Receipt (Bonnetje) Segmentation — YOLO11n-seg + Streamlit

A web app that detects receipts in photos and outlines them with a pixel-level
segmentation mask, then lets you download the straightened (deskewed) receipt.

* **Model:** YOLO11n-seg, 1 class (`bonnetje` = receipt), ~2.8 M parameters, ~6 MB
  — [`ciCic/yolo11n-seg-bonnetjes`](https://huggingface.co/ciCic/yolo11n-seg-bonnetjes) on Hugging Face
* **App:** [Streamlit](https://streamlit.io) · runs on CPU, no GPU needed

## Features

* Upload one or many images (JPG, PNG, WEBP, BMP) or use the **camera**
* Adjustable confidence / IoU / image-size settings
* Annotated image with masks, plus a table (confidence, box, mask area)
* Extracts each receipt as a straightened crop (perspective-corrected), downloadable as PNG
* Handles phone-photo EXIF rotation and downsizes very large images automatically

## Project structure

```
├── app.py                  # Streamlit UI
├── inference.py            # model loading, prediction, deskew helpers
├── requirements.txt        # Python dependencies (used by Streamlit Cloud)
├── packages.txt            # system libs for OpenCV on Streamlit Cloud
├── .streamlit/config.toml  # server / theme settings
├── models/                 # optional: put your own best.pt here
├── notebooks/              # original Colab training notebook
├── tests/                  # offline unit tests (pytest)
└── .github/workflows/ci.yml
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The first start downloads the ~6 MB weights from
Hugging Face and caches them.

Run the tests (no internet or GPU needed):

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Publish to GitHub

```bash
git init
git add .
git commit -m "Receipt segmentation Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud (free)

1. Push the repo to GitHub (above). The repo must be public, or you must grant Streamlit access.
2. Go to <https://share.streamlit.io> → **Create app** → pick your repo and branch `main`.
3. Set **Main file path** to `app.py`.
4. Open **Advanced settings** → choose **Python 3.12**.
5. Click **Deploy**. The first build takes several minutes (PyTorch is large).

`packages.txt` installs the system libraries OpenCV needs; no other setup is required.

## Using your own weights

Copy your trained checkpoint to `models/best.pt` (commit it; ~6 MB is fine for GitHub).
It takes priority over the Hugging Face download. You can also set the environment
variable `MODEL_PATH`, or `HF_REPO_ID` / `HF_FILENAME` to point at a different Hub model.

> **About the training notebook:** the notebook fine-tunes on *COCO128-seg*, which has
> 80 classes, while this model has 1 (`bonnetje`). Ultralytics skips every label it can't
> map, so that run learns nothing (validation mAP ≈ 0). This app therefore uses the original
> Hugging Face model. To really fine-tune, use a **receipt** segmentation dataset
> with a single class `0: bonnetje` and polygon labels.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ImportError: libGL.so.1` | Make sure `packages.txt` is in the repo root (it is included). |
| Build fails / runs out of disk or memory on Streamlit Cloud | Default PyTorch from PyPI ships multi-GB CUDA libraries. Put these **two lines at the top** of `requirements.txt` to use the small CPU build: `--extra-index-url https://download.pytorch.org/whl/cpu` then `torch==2.6.0+cpu` and `torchvision==0.21.0+cpu` (each on its own line). Or deploy on Hugging Face Spaces / Render / Docker. |
| "Could not load the model" | First run needs internet access to Hugging Face, or provide `models/best.pt`. |
| No detections | Lower the confidence slider; make sure the receipt is clearly visible. |

## License

MIT for the app code (see `LICENSE`). Ultralytics is AGPL-3.0; the model weights follow
the license on their Hugging Face page. Check both before commercial use.
