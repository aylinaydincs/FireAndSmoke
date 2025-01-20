import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel
from data_preprocessing import get_data_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(yaml_path, num_epochs=5, lr=1e-4, batch_size=32):
    train_loader, val_loader = get_data_loaders(yaml_path, batch_size)

    # Load CLIP Model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.train()

    # Add classification head
    classifier = nn.Linear(model.visual_projection.out_features, 3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = classifier(model.get_image_features(images))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = classifier(model.get_image_features(images))
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "fire_smoke_clip_v3.pth")
    print("Model saved!")


if __name__ == "__main__":
    train_model("data/data.yaml", num_epochs=15, batch_size=256, lr=1e-4)
