import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# =========================================
# DATASET PATH
# =========================================

dataset_path = "leapGestRecog"

# =========================================
# IMAGE SETTINGS
# =========================================

IMG_SIZE = 64

data = []
labels = []

# =========================================
# LOAD DATASET
# =========================================

print("Loading Dataset...")

for subject_folder in os.listdir(dataset_path):

    subject_path = os.path.join(dataset_path, subject_folder)

    if not os.path.isdir(subject_path):
        continue

    for gesture_folder in os.listdir(subject_path):

        gesture_path = os.path.join(subject_path, gesture_folder)

        if not os.path.isdir(gesture_path):
            continue

        # REDUCED IMAGES FOR FAST TRAINING
        for image_name in os.listdir(gesture_path)[:20]:

            image_path = os.path.join(gesture_path, image_name)

            try:
                image = cv2.imread(image_path)

                image = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY
                )

                image = cv2.resize(
                    image,
                    (IMG_SIZE, IMG_SIZE)
                )

                image = image.flatten()

                data.append(image)

                labels.append(gesture_folder)

            except:
                pass

# =========================================
# CONVERT TO NUMPY
# =========================================

X = np.array(data)

y = np.array(labels)

print("Images Loaded :", len(X))

# =========================================
# SCALE FEATURES
# =========================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# TRAIN SVM MODEL
# =========================================

print("Training Model...")

model = SVC(
    kernel='linear',
    max_iter=1000
)

model.fit(X_train, y_train)

print("Model Training Complete")

# =========================================
# PREDICTIONS
# =========================================

y_pred = model.predict(X_test)

# =========================================
# ACCURACY
# =========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", accuracy)

# =========================================
# CLASSIFICATION REPORT
# =========================================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))

plt.imshow(cm, cmap='Blues')

plt.title("Confusion Matrix")

plt.colorbar()

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

# =========================================
# SAMPLE PREDICTIONS
# =========================================

plt.figure(figsize=(12, 8))

for i in range(6):

    plt.subplot(2, 3, i + 1)

    image = X_test[i].reshape(
        IMG_SIZE,
        IMG_SIZE
    )

    plt.imshow(image, cmap='gray')

    plt.title(
        f"P:{y_pred[i]}\nA:{y_test[i]}"
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig("predictions.png")

plt.show()