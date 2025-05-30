import os
import cv2
import numpy as np
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import face_recognition


def load_images_and_labels(dataset_dir):
    embeddings = []
    labels = []

    print(f"Loading images from: {dataset_dir}")
    for person_name in os.listdir(dataset_dir):
        person_folder = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"Processing person images: {person_name}")
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)

            if len(encodings) == 0:
                print(f"No face in photo: {image_path}")
                continue

            embeddings.append(encodings[0])
            labels.append(person_name)

    print(f"extracted {len(embeddings)} embedding")
    return np.array(embeddings), np.array(labels)


def train_model(embeddings, labels):
    encoder = LabelEncoder()
    labels_encoded = encoder.fit_transform(labels)

    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels_encoded)

    return model, encoder


def save_model(model, encoder):
    dump(model, "face_recognition_model.joblib")
    dump(encoder, "label_encoder.joblib")
    print("The model and encoder label have been saved.")


def predict_images_in_folder(test_dir, model, encoder):
    print(f"All images are being tested in: {test_dir}")
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png",".webp"))]

    for image_name in image_files:
        image_path = os.path.join(test_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load image: {image_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        if len(encodings) == 0:
            print(f"Error: No face in the picture: {image_name}")
            continue
            
        for (box, encoding) in zip(boxes, encodings):
            probs = model.predict_proba([encoding])[0]
            pred_label = encoder.inverse_transform([np.argmax(probs)])[0]
            confidence = np.max(probs) * 100

            top, right, bottom, left = box
            cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 255), 2)

            label_text = f"{pred_label} ({confidence:.1f}%)"

            face_height = bottom - top
            font_scale = min(max(face_height / 240.0, 0.5), 1.0)
            font_thickness = max(int(font_scale * 2), 1)

            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                                  font_thickness)
            label_y = top - 10
            if label_y - text_height < 0:
                label_y = bottom + text_height + 10

            cv2.rectangle(image, (left, label_y - text_height - 5), (left + text_width, label_y + baseline), (0, 0, 0),-1)

            cv2.putText(image, label_text, (left, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        display = cv2.resize(image, (800, 600)) if image.shape[1] > 800 else image

        print(f"[-] {image_name} -> has been identified {len(encodings)} Face/Faces")
        cv2.imshow("Prediction", display)

        key = cv2.waitKey(0)
        if key == ord('q'):
            print(" Show has ended.")
            break
        elif key == ord('n'):
            continue
        else:
            continue

    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = input("Choose: (train/test):").strip()

    if mode == "train":
        print("Start training")
        dataset_path = "D:/Coding/python/Face_Identification/face_identification/train"
        embeddings, labels = load_images_and_labels(dataset_path)
        if len(embeddings) == 0:
            print("There is not enough data for training.")
        else:
            model, encoder = train_model(embeddings, labels)
            save_model(model, encoder)

    elif mode == "test":
        print("Test mode is enabled")
        model = load("face_recognition_model.joblib")
        encoder = load("label_encoder.joblib")
        test_dir = "D:/Coding/python/Face_Identification/face_identification/test2"
        predict_images_in_folder(test_dir, model, encoder)

    else:
        print("Incorrect choice. Type 'train' or 'test'.")
