import cv2
import face_recognition
import numpy as np
import pickle

EMBEDDING_FILE = "face_embeddings.pkl"
TOLERANCE = 0.7
PROCESS_EVERY_N_FRAMES = 30


def load_embeddings():
    with open(EMBEDDING_FILE, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['labels']


def recognize_from_video(video_path, known_embeddings, known_labels):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    results = []

    if not cap.isOpened():
        print("Failed to open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, boxes)

            results = []
            for (box, encoding) in zip(boxes, encodings):
                distances = face_recognition.face_distance(known_embeddings, encoding)
                best_match_index = np.argmin(distances)
                best_distance = distances[best_match_index]

                if best_distance < TOLERANCE:
                    name = known_labels[best_match_index]
                    confidence = max(0, 100 * (1 - best_distance ** 1.5))
                else:
                    name = "Unknown"
                    confidence = 0

                results.append((box, name, confidence))

        for box, name, confidence in results:
            top, right, bottom, left = [v * 4 for v in box]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{name} ({confidence:.1f}%)" if name != "Unknown" else name
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Face Recognition Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    embeddings, labels = load_embeddings()
    video_path = "D:/Coding/python/Face_Identification/face_identification/test2/Cristiano_Ronaldo.mp4"
    recognize_from_video(video_path, embeddings, labels)
