import os
import csv

def generate_csv(root_dir, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path_label'])

        for label_folder in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_folder)
            if not os.path.isdir(label_path):
                continue

            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_folder, img_name)
                label = label_folder
                writer.writerow([f"{img_path},{label}"])

    print("The file was created successfully")

generate_csv(
    root_dir='D:/Coding/python/Face_Identification/face_identification/train',
    output_csv_path='D:/Coding/python/Face_Identification/face_identification/trainset.csv'
)
