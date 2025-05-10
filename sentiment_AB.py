#chamisa Edmo 
#cv homework 4
#sentiment analysis with sys a and sys b


#imports
import os
import cv2
import numpy as np
import zipfile
import requests
import io
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report

#more imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
import time #this is for keeping track of time on run



#path
DATA_DIR = "/Users/chamisaedmo//Desktop/Spring_25/eecs841_CV/CE_HW4_CV/dataset"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32

#load in images and organize by file name
#this block is shared between systems a and b 
def load_images(data_dir):
    print(" Loading images...")
    images = []
    labels = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            label = 0 if filename.lower().startswith("happy") else 1
            filepath = os.path.join(data_dir, filename)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
    print(f" Loaded {len(images)} images from '{data_dir}'")
    print(f"   Happy: {sum(l == 0 for l in labels)} | Angry: {sum(l == 1 for l in labels)}")
    return np.array(images), np.array(labels)

#===== System A ======
#apply hog feature extractor
def extract_hog_features(images):
    print(" Extracting HoG features...")
    features = []
    for i, img in enumerate(images):
        feat = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        features.append(feat)
        if i % 500 == 0:
            print(f"  ...processed {i}/{len(images)} images")
    print(" HoG feature extraction complete.")
    return np.array(features)


#train it up based on hw directions
#this function performs 5-fold cross validation on all combos 
#and prints out he best parameter combo and eval metrics
def train_and_evaluate(X, y, label="System"):
    print(f"Splitting dataset and tuning SVM for {label}...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #define grid of hyperparameters
    param_grid = {
        'C': [0.1, 1],
        'gamma': ['scale', 'auto'],
        'kernel': ['linear', 'rbf', 'poly']
    }

    #use GridSearchCV to search for best combination
    grid = GridSearchCV(svm.SVC(), param_grid, cv=5, verbose=3, n_jobs=-1)
    start = time.time()
    grid.fit(X_train, y_train)
    print("⏱️ Grid search time:", round(time.time() - start, 2), "seconds")

    print(f"Best Params for {label}:", grid.best_params_)

    y_train_pred = grid.predict(X_train)
    y_test_pred = grid.predict(X_test)

    print(f"{label} Training Accuracy:", accuracy_score(y_train, y_train_pred))
    print(f"{label} Testing Accuracy:", accuracy_score(y_test, y_test_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=["Happy", "Angry"]))


#====== System B ======
class EmotionDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.labels = [0 if f.lower().startswith("happy") else 1 for f in self.image_files]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

#feature extraction for system b
def extract_deep_features(dataloader, model):
    print("Extracting your deep features...")
    model.eval()
    features_list, labels_list = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            outputs = model(images).squeeze()
            features_list.append(outputs.cpu().numpy())
            labels_list.extend(labels.numpy())

    return np.vstack(features_list), np.array(labels_list)

#run system b
def run_system_b():
    print("\nStarting System_B with ResNet18 feature extractor")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = EmotionDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load pre-trained ResNet18
    resnet = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(resnet.children())[:-1])  # remove final FC layer
    model.to(DEVICE)

    features, labels = extract_deep_features(dataloader, model)
    print(f"Extracted {features.shape[0]} deep feature vectors")
    train_and_evaluate(features, labels, label="System_B")


def main():
    print("Running System_A and Running System_B...")
    
    #=== System A ===
    print("\n=== SYSTEM A ===")
    images, labels = load_images(DATA_DIR)
    hog_features = extract_hog_features(images)
    train_and_evaluate(hog_features, labels, label="System_A")

    #=== System B ===
    print("\n=== SYSTEM B ===")
    run_system_b()

    print("Donezo!")

if __name__ == "__main__":
    main()