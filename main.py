import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

class FireSmokeDetector:
    def __init__(self):
        # Load the CLIP model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        state_dict = torch.load("fire_smoke_clip_v3.pth", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def predict(self, image_path):
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            text=["fire", "smoke", "no fire/smoke"], 
            images=image, 
            return_tensors="pt", 
            padding=True
        )

        # Get predictions
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

        # Return predictions
        labels = ["fire", "smoke", "no fire/smoke"]
        predictions = {labels[i]: float(probs[0][i]) for i in range(len(labels))}
        return predictions


from fastapi import FastAPI, UploadFile, File

app = FastAPI()
model = FireSmokeDetector()

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Save the uploaded file
    with open(file.filename, "wb") as buffer:
        buffer.write(await file.read())

    # Predict using the model
    predictions = model.predict(file.filename)
    return {"predictions": predictions}
