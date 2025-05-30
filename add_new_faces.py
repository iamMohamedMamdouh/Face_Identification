import os
import cv2
import numpy as np
import face_recognition
import pickle

EMBEDDING_FILE = "face_embeddings.pkl"


def extract_embeddings_from_folder(dataset_dir):
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
    return embeddings, labels


def save_embeddings(embeddings, labels):
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, 'rb') as f:
            existing = pickle.load(f)

        existing_embeddings = np.array(existing['embeddings'])
        existing_labels = np.array(existing['labels'])
        new_embeddings = np.array(embeddings)
        new_labels = np.array(labels)

        embeddings = np.concatenate([existing_embeddings, new_embeddings])
        labels = np.concatenate([existing_labels, new_labels])
    else:

        embeddings = np.array(embeddings)
        labels = np.array(labels)

    with open(EMBEDDING_FILE, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, f)
    print("The model and encoder label have been saved.")


def load_embeddings():
    if not os.path.exists(EMBEDDING_FILE):
        print("No embeddings file found.")
        return [], []
    with open(EMBEDDING_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['labels']


def predict_images_in_folder(test_dir, known_embeddings, known_labels, tolerance=0.5):
    print(f"Testing images from: {test_dir}")
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]

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
            print(f"Error: No face in image: {image_name}")
            continue

        for (box, encoding) in zip(boxes, encodings):
            distances = face_recognition.face_distance(known_embeddings, encoding)
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            tolerance = 0.7  

            if best_distance < tolerance:
                name = known_labels[best_match_index]
                confidence = max(0, 100 * (1 - best_distance ** 1.5))
            else:
                name = "Unknown"
                confidence = 0
            if distances[best_match_index] < tolerance:
                name = known_labels[best_match_index]
                confidence = (1.0 - distances[best_match_index]) * 100  

            top, right, bottom, left = box
            cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 255), 2)

            label_text = f"{name} ({confidence:.1f}%)" if name != "Unknown" else name

            face_height = bottom - top
            font_scale = min(max(face_height / 240.0, 0.5), 1.0)
            font_thickness = max(int(font_scale * 2), 1)

            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                                  font_thickness)
            label_y = top - 10
            if label_y - text_height < 0:
                label_y = bottom + text_height + 10

            cv2.rectangle(image, (left, label_y - text_height - 5), (left + text_width, label_y + baseline), (0, 0, 0),
                          -1)
            cv2.putText(image, label_text, (left, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        font_thickness)

        display = cv2.resize(image, (800, 600)) if image.shape[1] > 800 else image
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


def predict_video(video_path, known_embeddings, known_labels, tolerance=0.6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps) if fps > 0 else 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for (box, encoding) in zip(boxes, encodings):
            distances = face_recognition.face_distance(known_embeddings, encoding)
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]

            if best_distance < tolerance:
                name = known_labels[best_match_index]
                confidence = max(0, 100 * (1 - best_distance ** 1.5))
            else:
                name = "Unknown"
                confidence = 0

            top, right, bottom, left = box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{name} ({confidence:.1f}%)" if name != "Unknown" else name

            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        display = cv2.resize(frame, (800, 600)) if frame.shape[1] > 800 else frame
        cv2.imshow("Face Recognition Video", display)

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mode = input("Choose mode (train / test / add / video): ").strip().lower()
    if mode == "train":
        print("Start training")
        dataset_path = "D:/Coding/python/Face_Identification/face_identification/train"
        embeddings, labels = extract_embeddings_from_folder(dataset_path)
        if embeddings:
            save_embeddings(embeddings, labels) 
        else:
            print("There is not enough data for training.")


    elif mode == "add":
        print("Add new faces mode")
        dataset_path = "D:/Coding/python/Face_Identification/face_identification/train/new_faces"
        new_embeddings, new_labels = extract_embeddings_from_folder(dataset_path)
        if not new_embeddings:
            print("No new data found.")
        else:
            try:
                old_embeddings, old_labels = load_embeddings()
                old_embeddings = np.array(old_embeddings)
                old_labels = np.array(old_labels)
                new_embeddings = np.array(new_embeddings)
                new_labels = np.array(new_labels)

                embeddings = np.concatenate([old_embeddings, new_embeddings])

                labels = np.concatenate([old_labels, new_labels])
            except FileNotFoundError:
                embeddings, labels = np.array(new_embeddings), np.array(new_labels)
            save_embeddings(embeddings, labels)
            print("New faces added successfully.")

    elif mode == "video":
        print("Video mode enabled")
        video_path = "D:/Coding/python/Face_Identification/face_identification/test2/Cristiano_Ronaldo.mp4"
        embeddings, labels = load_embeddings()
        if embeddings is not None and embeddings.size > 0:
            predict_video(video_path, embeddings, labels)
        else:
            print("No embeddings found. Please run in 'train' or 'add' mode first.")

    elif mode == "test":
        print("Test mode is enabled")
        test_dir = "D:/Coding/python/Face_Identification/face_identification/test2"
        embeddings, labels = load_embeddings()
        if embeddings is not None and embeddings.size > 0:
            predict_images_in_folder(test_dir, embeddings, labels)
        else:
            print("No embeddings found. Please run in 'train' or 'add' mode first.")

    else:
        print("Invalid mode. Choose: train / test / add.")
