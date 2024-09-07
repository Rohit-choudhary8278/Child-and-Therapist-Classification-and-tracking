# Child-and-Therapist-Classification-and-tracking
## Overview

This assignment focuses on analyzing video files by leveraging object detection and tracking techniques. The primary goal is to detect objects (child and therapist) in each frame of a video and track their movements across frames. The analysis is performed using the YOLO (You Only Look Once) model for object detection and the DeepSORT (Simple Online and Realtime Tracking with a Deep Association Metric) algorithm for tracking.

## Workflow

The following steps outline the logic and workflow of the video processing pipeline:

1. **Model Loading**:
    - A YOLO model is loaded for object detection. The model is pre-trained on a dataset to recognize specific classes of objects.
    - A ResNet50 model is used for feature extraction, helping the tracking algorithm distinguish different objects.

2. **Tracker Initialization**:
    - The DeepSORT tracker is initialized with default parameters. This tracker uses both motion and appearance information to associate detected objects across frames, maintaining their unique IDs.

3. **Video Reading**:
    - The program opens a video file using OpenCV and retrieves its properties, such as frame width, height, and frames per second (FPS).

4. **Frame-by-Frame Processing**:
    - For each frame of the video:
        - **Object Detection**: The YOLO model detects objects within the frame, returning bounding box coordinates, confidence scores, and class IDs.
        - **Feature Extraction**: For each detected object, the ResNet model extracts appearance features, which help in distinguishing between similar objects.
        - **Tracking Update**: The DeepSORT tracker is updated with new detections and corresponding features to maintain consistent object IDs across frames.
        - **Drawing Results**: The program draws bounding boxes around detected objects, along with their unique IDs and labels (object class names) on the frame.

5. **Video Writing**:
    - The processed frames are written to a new video file using OpenCV. This output video includes the bounding boxes and labels for detected objects.

6. **Logging**:
    - The names of processed video files are recorded in a log file (`progress.log`). This log helps avoid reprocessing files during subsequent runs of the script.

## How to Reproduce Results

Follow these steps to reproduce the results:

### 1. Set Up the Environment

- Install the required dependencies:
    ```sh
    pip install opencv-python torch torchvision ultralytics deep-sort-realtime
    ```
  
- Ensure that your environment has GPU support for PyTorch to accelerate model inference.

### 2. Prepare the Model

- Download  the YOLO model file (`model c.pt`) in the `model_path` specified in the script.

### 3. Organize Input Data

- Place the input videos to be processed in the `input_dir` directory (`/kaggle/input/test-videos`).
- Make sure the directory exists and contains videos in `.mp4` format.

### 4. Run the Script

- Execute the script to start processing the videos:
    ```sh
    python main..ipynb
    ```

- The script will read each video from the `input_dir`, process it, and save the output in the `output_dir` directory (`/kaggle/working/output`).

### 5. Check Results

- The processed videos with bounding boxes and labels will be saved in the `output_dir`.
- The `progress.log` file will contain the names of videos that have been processed. You can check this log to confirm which files have been completed and to avoid duplicates in future runs.

## Key Components

### YOLO Model

- The YOLO (You Only Look Once) model is a popular real-time object detection system known for its speed and accuracy. It detects objects in images or video frames by dividing the image into grids and predicting bounding boxes and class probabilities.

### DeepSORT Tracker

- DeepSORT (Deep Simple Online and Realtime Tracking) is an extension of the SORT algorithm. It uses appearance descriptors (features) extracted by a neural network, in addition to motion information, to associate detected objects between frames. This helps in maintaining unique identities for each object over time.

## Analyzing the Model Predictions

1. **Confidence Threshold**:
    - The model's predictions include confidence scores. Only detections with confidence scores above a certain threshold are considered for tracking to avoid false positives.

2. **Bounding Box and Class ID**:
    - For each detected object, the bounding box coordinates and class IDs are obtained. This information is crucial for both visualization and accurate tracking.

3. **Feature Extraction**:
    - Features extracted from detected objects using ResNet50 are used to distinguish objects of the same class but with different identities (e.g., multiple people in the same frame).

4. **Tracking with DeepSORT**:
    - The DeepSORT tracker updates the object trajectories using both motion (Kalman filter) and appearance (feature descriptors) data, ensuring robust tracking even with occlusions or similar-looking objects.

## Test Video Output link
   https://drive.google.com/drive/folders/1kzT8Z2tbdZo9syKmkY-_Yy7__8r83cr2?usp=sharing
   
