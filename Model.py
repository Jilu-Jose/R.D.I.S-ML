import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import requests
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib


model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


image_dir = "images"
os.makedirs(image_dir, exist_ok=True)

def download_image(img_url, img_path):
    """Download image from URL and save it locally."""
    try:
        response = requests.get(img_url, stream=True)
        if response.status_code == 200:
            with open(img_path, "wb") as file:
                file.write(response.content)
        else:
            print(f"Failed to download image: {img_url}")
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")

def extract_features(image_path):
    """Extract features from an image using ResNet50."""
    if not os.path.exists(image_path):
        print(f"Error: Image not found - {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image - {image_path}")
        return None

    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()


csv_path = "resized_dataset.csv"
df = pd.read_csv(csv_path)


feature_list = []
labels = []
for index, row in df.iterrows():
    img_path = os.path.join(image_dir, f"{row['id']}.jpg")
    if not os.path.exists(img_path):
        download_image(row['img'], img_path)

    features = extract_features(img_path)
    if features is not None:
        feature_list.append(features)
        labels.append(row['name'])


if len(feature_list) == 0:
    print("Error: No valid images found for training.")
    exit()

X = np.array(feature_list)
y = np.array(labels)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


classifier = SVC(kernel='linear', probability=True)
classifier.fit(X, y_encoded)


joblib.dump(classifier, "face_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

def predict_person(query_image_path):
    """Predict person details from an image."""
    query_features = extract_features(query_image_path)
    if query_features is None:
        return None

    query_features = query_features.reshape(1, -1)
    prediction = classifier.predict(query_features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    person = df[df['name'] == predicted_label].to_dict(orient='records')[0]
    return "\n Personal-Details: \n" f"""
    Name: {person['name']}
    Id: {person['id']}
    D.O.B: {person['dob']}
    Age: {person['age']}
    Blood-Group: {person['bloodGroup']}
    """


query_image = "/content/images_100/11.jpg"
person_details = predict_person(query_image)
if person_details:
    print(person_details)
else:
    print("No match found.")