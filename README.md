# MarsVision AI — Terrain Telemetry & Segmentation

A Flask web application powered by a **DeepLabV3+ (`ResNet50`)** PyTorch model (`deeplabv3plus_ai4mars.pt`) for pixel-level Mars rover terrain
segmentation (**Soil**, **Bedrock**, **Sand**, **Big Rock**).

## Key Features
- **Terrain Segmentation & Overlay:** Generates segmentation masks, blended overlays, and before/after comparison sliders.
- **Telemetry Dashboard:** Shows terrain composition percentages, dominant terrain summary, and traversal advisory cards.
- **Sample Image Loader:** Quick one-click sample selection from `static/samples/` alongside custom image upload (`.jpg`, `.png`, `.jfif`, `.webp`).

## Project Structure
- `server.py` — Flask backend and PyTorch inference pipeline
- `templates/index.html` — Frontend UI and interactive telemetry visualization
- `static/samples/` — Sample rover images (`mars pic.jpg`) for quick testing
- `deeplabv3plus_ai4mars.pt` — Trained model weights (place in the project root directory).
- [Download deeplabv3plus_ai4mars.pt](https://github.com/shambhaviyaduvanshi/marsVision-AI/releases/download/v1.0/deeplabv3plus_ai4mars.pt)

## Run Locally
```bash
pip install flask numpy pillow torch torchvision segmentation-models-pytorch
python server.py

**Deployment Note**
Running a PyTorch DeepLabV3+ (ResNet50) model requires sufficient memory and compute, which is often restricted or paid on
free hosting tiers. This project is set up for local execution and higher-memory environments.
