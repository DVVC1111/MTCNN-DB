import os
import cv2
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
import numpy as np
import requests
from io import BytesIO
import boto3
import botocore

import mysql.connector
from mysql.connector import Error

def fix_image_orientation(image):
    return ImageOps.exif_transpose(image)

def insert_face_count_data(image_name, face_count):
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="David910139",
            database="face_detection"
        )
        cursor = connection.cursor()
        query = "INSERT INTO face_counts (image_name, face_count) VALUES (%s, %s)"
        cursor.execute(query, (image_name, face_count))
        connection.commit()
        print(f"Face count data for image {image_name} inserted successfully")

    except Error as e:
        print(f"Error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def detect_faces(base_url, folder_name, output_folder_name, detector, img_numbers):
    s3_client = boto3.client("s3", region_name="ap-southeast-2", config=botocore.client.Config(signature_version=botocore.UNSIGNED))

    for num in img_numbers:
        filename = f"img{num}.jpg"
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            image_url = f"{base_url}/{folder_name}/{filename}"

            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            image = fix_image_orientation(image)

            image = image.convert("RGB")

            bboxes, _ = detector.detect(image)

            image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            face_count = 0
            if bboxes is not None:
                face_count = len(bboxes)
                for bbox in bboxes:
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])
                    cv2.rectangle(image_cv2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            label = f"Faces detected: {face_count}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            text_x, text_y = 10, 30
            cv2.putText(image_cv2, label, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)

            success, buffer = cv2.imencode('.jpg', image_cv2)
            if not success:
                print(f"Failed to encode image {filename}")
                continue

            output_filename = f"output_{filename.split('.')[0]}_facecount{face_count}.jpg"

            print(f"Uploading {output_filename} to S3...")
            s3_client.put_object(Bucket="mtcnn", Key=f"{output_folder_name}/{output_filename}", Body=BytesIO(buffer))
            print(f"Upload completed for {output_filename}")

            insert_face_count_data(filename, face_count)

            output_image_url = f"{base_url}/{output_folder_name}/{output_filename}"
            print(f"Output image URL: {output_image_url}")

    print("Face detection complete.")
    
def main():
    base_url = "https://mtcnn.s3.ap-southeast-2.amazonaws.com"
    folder_name = "image"
    output_folder_name = "output_image"
    img_numbers = [1, 2, 3, 4, 5, 6]

    detector = MTCNN(thresholds=[0.65, 0.75, 0.75])

    detect_faces(base_url, folder_name, output_folder_name, detector, img_numbers)

if __name__ == "__main__":
    main()