
# **Fire and Smoke Detection Using Vision-Language Models**

This project implements a Vision-Language Model (VLM)-based system to classify images into three categories: **fire**, **smoke**, or **no fire/smoke** (no fire or smoke). The solution leverages OpenAI's CLIP model and the Roboflow Fire and Smoke Detection dataset, formatted in YOLO annotation style. The project also includes a FastAPI-based API for deployment and inference.

---

## **Overview**

- **Vision-Language Model**: OpenAI's CLIP (Contrastive Language-Image Pretraining) is used for feature extraction and classification.
- **Dataset**: Roboflow's Fire and Smoke Detection dataset with YOLO-format annotations.
- **Deployment**: A FastAPI application serves predictions via an HTTP endpoint.
- **Use Case**: Early detection of fire and smoke in images for safety and disaster prevention.

---

## **Why CLIP?**

### **Key Reasons**
1. **Multi-Modal Learning**:
   - CLIP is a pre-trained Vision-Language Model capable of understanding visual content in the context of text prompts.
   - It eliminates the need for extensive fine-tuning by leveraging textual descriptions such as "fire," "smoke," and "no fire/smoke."

2. **Zero-Shot and Few-Shot Learning**:
   - CLIP performs well with minimal labeled data.
   - For this project, only limited fine-tuning was required to adapt CLIP to the fire and smoke classification problem.

3. **Scalability and Generalization**:
   - By using a model pre-trained on a diverse range of image-text pairs, CLIP generalizes better across various environmental conditions (indoor, outdoor, varying lighting).

4. **Efficiency**:
   - CLIP’s architecture (ViT-based for images) ensures high accuracy with manageable computational resources, making it suitable for deployment in real-world scenarios.

---

## **Methodology**

1. **Data Preprocessing**:
   - The dataset was formatted in YOLO annotation style, with three classes: **fire**, **smoke**, and **no fire/smoke**.
   - Images were resized, normalized, and converted into tensors for compatibility with CLIP’s ViT backbone.

2. **Model Training**:
   - CLIP’s pre-trained model was fine-tuned on the Roboflow dataset.
   - A classification head (linear layer) was added to classify the image into one of the three categories.
   - The model was trained using PyTorch with CrossEntropy loss.

3. **Evaluation**:
   - The trained model was evaluated using accuracy, loss, and F1-score metrics on the validation set.
   - Validation was performed to ensure robust performance across unseen images.

4. **Deployment**:
   - A FastAPI-based API was developed to allow users to upload images and receive predictions in real time.
   - Predictions return probabilities for each class (fire, smoke, no fire/smoke).

---

## **Setup and Installation**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd <repository-directory>
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Dataset Preparation**
1. Download the dataset from Roboflow:
   [Fire and Smoke Detection Dataset](https://universe.roboflow.com/middle-east-tech-university/fire-and-smoke-detection-hiwia).
2. Ensure the dataset follows this structure:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   ├── labels/
   ├── valid/
   │   ├── images/
   │   ├── labels/
   ├── test/
   │   ├── images/
   │   ├── labels/
   └── data.yaml
   ```
3. Update `data.yaml`:
   ```yaml
   nc: 3
   names: ['no fire/smoke', 'smoke', 'fire']
   ```

---

## **Model Training**

Train the model using:
```bash
python train.py
```

### **Parameters**
- `yaml_path`: Path to the dataset YAML file (default: `data/data.yaml`).
- `num_epochs`: Number of training epochs (default: 15).
- `batch_size`: Batch size for training (default: 256).
- `lr`: Learning rate (default: 1e-4).

The trained model is saved as `fire_smoke_clip_v3.pth`.

---

## **API Deployment**

Run the FastAPI application:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### **API Endpoints**
1. **Health Check**
   - **Endpoint**: `GET /`
   - **Description**: Verifies the API is running.

2. **Classify Image**
   - **Endpoint**: `POST /classify`
   - **Description**: Upload an image to classify it as **fire**, **smoke**, or **neutral**.
   - **Request Example**:
     ```bash
     curl -X POST -F "file=@path_to_image.jpg" http://localhost:8000/classify
     ```
   - **Response Example**:
     ```json
     {
       "predictions": {
         "fire": 0.85,
         "smoke": 0.10,
         "no fire/smoke": 0.05
       }
     }
     ```

---

## **File Structure**

```
.
├── data_preprocessing.py   # Dataset loading and preprocessing
├── main.py                 # FastAPI application for inference
├── train.py                # Model training script
├── requirements.txt        # Dependencies
├── dataset/                # Dataset directory (not included in repo)
│   └── data.yaml           # Dataset configuration
├── fire_smoke_clip_v2.pth  # Trained model file (generated after training)
```

---

## **Acknowledgments**

- **OpenAI**: For providing the CLIP model.
- **Roboflow**: For the Fire and Smoke Detection dataset.
- **YOLO Format**: For its efficient annotation structure.

---

## **License**

The dataset is licensed under **CC BY 4.0**. The project code is free to use for educational and research purposes.
