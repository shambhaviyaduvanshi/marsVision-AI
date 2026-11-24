import base64
import io
import os
import time
from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.functional as TF

app = Flask(__name__)

WEIGHTS_PATH = "deeplabv3plus_ai4mars.pt"
SAMPLES_DIR = os.path.join("static", "samples")
IMG_SIZE = (256, 256)
CLASSES = ["Soil", "Bedrock", "Sand", "Big Rock"]
HEX_COLORS = ["#c27b37", "#787882", "#ebbe5f", "#dc2d2d"]
COLORS = np.array(
    [
        [194, 123, 55],  # Soil
        [120, 120, 130],  # Bedrock
        [235, 190, 95],  # Sand
        [220, 45, 45],  # Big Rock
    ],
    dtype=np.uint8,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = smp.DeepLabV3Plus(
    encoder_name="resnet50", encoder_weights=None, classes=len(CLASSES)
).to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()


def img_to_base64(pil_img):
  buf = io.BytesIO()
  pil_img.save(buf, format="PNG")
  return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(
      "utf-8"
  )


def build_advisory(pct_map):
  rock_total = pct_map.get("Big Rock", 0.0) + pct_map.get("Bedrock", 0.0)
  sand_pct = pct_map.get("Sand", 0.0)
  soil_pct = pct_map.get("Soil", 0.0)

  if pct_map.get("Big Rock", 0.0) >= 15.0 or rock_total >= 45.0:
    return {
        "level": "Caution",
        "title": "High Rocky / Bedrock Density",
        "note": (
            "Large rocks or bedrock detected—plan obstacle-aware rover pathing"
            " and check wheel clearance."
        ),
    }
  if sand_pct >= 80.0:
    return {
        "level": "Advisory",
        "title": "Predominantly Sandy Terrain",
        "note": (
            "High sand coverage—monitor traction and potential wheel slip on"
            " slopes."
        ),
    }
  if soil_pct >= 25.0:
    return {
        "level": "Nominal",
        "title": "Mixed Soil & Surface Regolith",
        "note": (
            "Balanced surface composition—suitable for standard rover traverse"
            " and surface sampling."
        ),
    }
  return {
      "level": "Nominal",
      "title": "Standard Rover Telemetry",
      "note": "Terrain segmented across 4 AI4MARS classes.",
  }


@app.route("/")
def home():
  return render_template("index.html")


@app.route("/samples")
def list_samples():
  files = []
  if os.path.isdir(SAMPLES_DIR):
    exts = (".jpg", ".jpeg", ".png", ".webp", ".jfif")
    for fname in sorted(os.listdir(SAMPLES_DIR)):
      if fname.lower().endswith(exts):
        files.append(fname)
  return jsonify({"samples": files[:6]})


@app.route("/predict", methods=["POST"])
def predict():
  if "image" not in request.files:
    return jsonify({"error": "No image uploaded"}), 400

  t0 = time.perf_counter()
  file = request.files["image"]
  alpha = float(request.form.get("alpha", 0.5))

  orig_pil = Image.open(file.stream).convert("RGB")
  orig_w, orig_h = orig_pil.size

  resized_pil = TF.resize(
      orig_pil, IMG_SIZE, interpolation=TF.InterpolationMode.BILINEAR
  )
  tensor = (
      TF.normalize(
          TF.to_tensor(resized_pil),
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225],
      )
      .unsqueeze(0)
      .to(device)
  )

  with torch.no_grad():
    pred_small = torch.argmax(model(tensor), dim=1).squeeze(0).cpu().numpy()

  pred_pil = TF.resize(
      Image.fromarray(pred_small.astype(np.uint8)),
      (orig_h, orig_w),
      interpolation=TF.InterpolationMode.NEAREST,
  )
  pred_mask = np.array(pred_pil)

  color_mask = COLORS[pred_mask]
  orig_np = np.array(orig_pil)
  overlay_np = (
      (1.0 - alpha) * orig_np.astype(np.float32)
      + alpha * color_mask.astype(np.float32)
  ).astype(np.uint8)

  total_px = pred_mask.size
  stats = []
  pct_map = {}
  dominant = {"name": CLASSES[0], "percent": 0.0}
  for idx, cname in enumerate(CLASSES):
    pct = round(float((np.sum(pred_mask == idx) / total_px) * 100.0), 1)
    pct_map[cname] = pct
    stats.append({"name": cname, "percent": pct, "color": HEX_COLORS[idx]})
    if pct >= dominant["percent"]:
      dominant = {"name": cname, "percent": pct}

  elapsed_ms = round((time.perf_counter() - t0) * 1000)

  return jsonify({
      "original": img_to_base64(orig_pil),
      "mask": img_to_base64(Image.fromarray(color_mask)),
      "overlay": img_to_base64(Image.fromarray(overlay_np)),
      "stats": stats,
      "dominant": dominant,
      "advisory": build_advisory(pct_map),
      "time_ms": elapsed_ms,
      "resolution": f"{orig_w}×{orig_h}",
  })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)