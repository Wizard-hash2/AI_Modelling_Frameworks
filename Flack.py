from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from torchvision import transforms

# 1. Load a pretrained PyTorch model
model = torch.load("resnet18_finetuned.pt", map_location="cpu")
model.eval()

# 2. Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Initialize Flask app
app = Flask(__name__)

# 4. Create /predict route
@app.route("/predict", methods=["POST"])
def predict():
    # 4.1 Read image bytes from POST body
    img_bytes = request.data
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # 4.2 Preprocess & batch dimension
    input_tensor = preprocess(image).unsqueeze(0)
    # 4.3 Model inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred_idx = outputs.max(1)
    # 4.4 Return class index as JSON
    return jsonify({"predicted_class": int(pred_idx.item())})

# 5. Run the app
if __name__ == "__main__":
    app.run(debug=True)
