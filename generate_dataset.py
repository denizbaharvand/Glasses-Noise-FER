"""
RAF-DB Noisy Dataset Generation
This script extracts the RAF-DB dataset, uses MediaPipe to detect facial landmarks,
applies Gaussian noise specifically to the eye regions to create a challenging 
variant of the dataset, and archives the result for later model training.
"""

# ==========================================
# 1. Environment Setup
# ==========================================
# Note: Uncomment the following lines if running directly in Google Colab.
# from google.colab import drive
# drive.mount('/content/drive')
# !pip uninstall -y numpy opencv-python mediapipe matplotlib
# !pip install numpy==1.23.5 opencv-python-headless==4.8.1.78 mediapipe==0.10.11 matplotlib==3.7.1

import os
import zipfile
import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt

# ==========================================
# 2. Dataset Extraction & Validation
# ==========================================
zip_path = '/content/drive/MyDrive/archive.zip'  # Path to the original RAF-DB zip archive
extract_path = '/content/RAF-DB'

# Extract the dataset if the zip file exists
if not os.path.exists(zip_path):
    print(f"Error: File {zip_path} not found. Please check the path.")
else:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction completed successfully!")

# Validate and print the directory structure (first 3 files per folder)
print("Dataset structure overview:")
for root, dirs, files in os.walk(extract_path):
    level = root.replace(extract_path, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 4 * (level + 1)
    for f in files[:3]:
        print(f'{subindent}{f}')

# ==========================================
# 3. Noisy Dataset Generation (Eye Region)
# ==========================================
# Initialize MediaPipe Face Detection modules
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def add_noise_to_eyes(image_path, output_path, noise_std=50, eye_size=50):
    """
    Detects faces in an image using MediaPipe and applies Gaussian noise 
    specifically to the bounded regions around the eyes.
    
    Args:
        image_path (str): Path to the original input image.
        output_path (str): Destination path to save the noisy image.
        noise_std (int): Standard deviation of the Gaussian noise.
        eye_size (int): Size of the square bounding box around each eye.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Convert BGR to RGB as required by MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize face detection model
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                # Extract relative keypoints for the eyes
                keypoints = detection.location_data.relative_keypoints
                left_eye = keypoints[0]
                right_eye = keypoints[1]

                # Map relative coordinates to absolute pixel values
                h, w, _ = image.shape
                left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
                right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

                # Define the square regions for both eyes
                left_eye_region = image[max(0, left_eye_y - eye_size//2):min(h, left_eye_y + eye_size//2),
                                        max(0, left_eye_x - eye_size//2):min(w, left_eye_x + eye_size//2)]
                right_eye_region = image[max(0, right_eye_y - eye_size//2):min(h, right_eye_y + eye_size//2),
                                         max(0, right_eye_x - eye_size//2):min(w, right_eye_x + eye_size//2)]

                # Generate and apply Gaussian noise to the extracted eye regions
                if left_eye_region.size > 0:
                    noise_left = np.random.normal(0, noise_std, left_eye_region.shape).astype(np.uint8)
                    left_eye_region[:] = cv2.add(left_eye_region, noise_left)
                    
                if right_eye_region.size > 0:
                    noise_right = np.random.normal(0, noise_std, right_eye_region.shape).astype(np.uint8)
                    right_eye_region[:] = cv2.add(right_eye_region, noise_right)

            # Save the modified image
            cv2.imwrite(output_path, image)
        else:
            # Save the original image if no face is detected
            cv2.imwrite(output_path, image)

# Define input/output directories for dataset generation
input_dir = '/content/RAF-DB/DATASET/train'  
output_dir = '/content/RAF-DB-noisy/train'
os.makedirs(output_dir, exist_ok=True)

# Process the dataset and track progress
total_images = sum(len(files) for _, _, files in os.walk(input_dir))
processed = 0

print("\nStarting noise generation process...")
for class_label in os.listdir(input_dir):
    class_path = os.path.join(input_dir, class_label)
    if not os.path.isdir(class_path):
        continue
        
    output_class_path = os.path.join(output_dir, class_label)
    os.makedirs(output_class_path, exist_ok=True)

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        output_path = os.path.join(output_class_path, img_name)
        
        # Apply noise (parameters can be tuned here)
        add_noise_to_eyes(img_path, output_path, noise_std=50, eye_size=50)
        
        processed += 1
        if processed % 1000 == 0 or processed == total_images:
            print(f"Progress: {processed}/{total_images} ({(processed/total_images)*100:.2f}%)")

# ==========================================
# 4. Data Visualization & Archiving
# ==========================================
# Visualize a sample before and after applying noise
sample_img_path = '/content/RAF-DB/DATASET/train/1/train_0001.jpg' 
sample_noisy_path = '/content/RAF-DB-noisy/train/1/train_0001.jpg'

original = cv2.imread(sample_img_path)
noisy = cv2.imread(sample_noisy_path)

if original is None or noisy is None:
    print("\nNotice: Sample images not found for visualization. Ensure paths are correct.")
else:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Noisy Eyes")
    plt.imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

# Archive the generated noisy dataset
output_zip = '/content/RAF-DB-noisy.zip'
if os.path.exists(output_dir):
    # Note: Colab specific bash commands for archiving
    print("\nArchiving the generated noisy dataset...")
    os.system(f'zip -r {output_zip} {output_dir}')
    os.system(f'cp {output_zip} /content/drive/MyDrive/RAF-DB-noisy.zip')
    print("Archived and saved the noisy dataset to Google Drive successfully.")
else:
    print(f"\nError: Noisy dataset directory {output_dir} not found.")
