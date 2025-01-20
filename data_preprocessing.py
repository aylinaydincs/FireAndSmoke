import os
import yaml
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, classes, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes = classes
        self.transform = transform

        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')

        # Load label
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        with open(label_path, 'r') as f:
            lines = f.readlines()

        labels = [list(map(float, line.strip().split())) for line in lines]
        class_ids = [int(line[0]) for line in labels]

        # Determine overall class (e.g., 0 for fire, 1 for smoke)
        overall_class = max(class_ids) if class_ids else 0  # Default to 'fire'

        if self.transform:
            image = self.transform(image)

        return image, overall_class


def get_data_loaders(yaml_path, batch_size=32):
    # Load YAML
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    classes = data['names']
    train_loader = DataLoader(
        YOLODataset(
            images_dir=data['train'],
            labels_dir=data['train'].replace('images', 'labels'),
            classes=classes,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        ),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        YOLODataset(
            images_dir=data['val'],
            labels_dir=data['val'].replace('images', 'labels'),
            classes=classes,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        ),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader
