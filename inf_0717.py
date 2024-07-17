import os
import cv2
import numpy as np
from openvino.runtime import Core, Tensor
from ultralytics import YOLO
import matplotlib.pyplot as plt
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 키포인트 추출 함수
def extract_keypoints_from_video(video_path, output_folder):
    logging.info(f"Starting keypoint extraction for video: {video_path}")
    model = YOLO('yolov8n-pose.pt')
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    previous_keypoints = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        keypoints = results[0].keypoints
        if keypoints is not None and len(keypoints) > 0:
            keypoints = keypoints[0].xy.cpu().numpy().reshape((17, 2))
            logging.info(f"Frame {frame_count}: raw keypoints shape {keypoints.shape}")
            previous_keypoints = keypoints
        elif previous_keypoints is not None:
            keypoints = previous_keypoints
        else:
            keypoints = np.zeros((17, 2))
        
        logging.info(f"Frame {frame_count}: final keypoints shape {keypoints.shape}")
        np.save(os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_frame_{frame_count:05d}.npy"), keypoints)
        
        frame_count += 1
    cap.release()
    logging.info(f"Finished keypoint extraction. Total frames processed: {frame_count}")

# 패딩 함수 수정
def pad_segment(segment, max_length):
    padded_segment = np.zeros((max_length, 17, 2))
    for i, frame in enumerate(segment):
        logging.info(f"Padding frame {i}: keypoints shape {frame.shape}")
        if i >= max_length:
            break
        if frame.shape == (17, 2):
            padded_segment[i] = frame
        else:
            logging.warning(f"Frame {i} shape {frame.shape} does not match expected shape (17, 2). Filling with zeros.")
            padded_segment[i] = np.zeros((17, 2))
    return padded_segment

# 이동 평균 함수 (변경 없음)
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# 이상 행동 감지 함수
def predict_anomaly(video_path):
    logging.info(f"Starting anomaly prediction for video: {video_path}")
    max_length = 119

    extract_keypoints_from_video(video_path, "temp_output")
    
    frames = [np.load(os.path.join("temp_output", f)) for f in sorted(os.listdir("temp_output")) if f.endswith(".npy")]
    logging.info(f"Loaded {len(frames)} frames for prediction")

    all_predictions = []
    step_size = max_length // 2
    for start in range(0, len(frames), step_size):
        segment = frames[start:start + max_length]
        logging.info(f"Processing segment starting at frame {start}, segment length: {len(segment)}")
        segment = pad_segment(segment, max_length)
        logging.info(f"Padded segment shape: {segment.shape}")

        segment = np.nan_to_num(segment).astype(np.float32)
        segment = segment.reshape(1, max_length, 17, 2).reshape(1, max_length, 34)

        infer_request = compiled_model.create_infer_request()
        input_tensor = Tensor(segment)
        infer_request.set_tensor(input_layer, input_tensor)
        infer_request.infer()
        predictions = infer_request.get_tensor(output_layer).data
        anomaly_scores = [pred for pred in predictions.flatten()]
        all_predictions.extend(anomaly_scores)
        logging.info(f"Processed segment. Current total predictions: {len(all_predictions)}")
    
    if len(all_predictions) > 0:  # Check if the list is not empty
        min_score = np.min(all_predictions)
        max_score = np.max(all_predictions)
        normalized_scores = [(score - min_score) / (max_score - min_score) * 100 for score in all_predictions]
        
        smooth_scores = moving_average(normalized_scores, window_size=5)
        logging.info(f"Finished prediction. Total smooth scores: {len(smooth_scores)}")
    else:
        smooth_scores = []
        logging.warning("No predictions were made. Check if keypoints were correctly extracted.")
    
    return np.array(smooth_scores)

# OpenVINO 모델 불러오기
logging.info("Loading OpenVINO model")
core = Core()
model_path = 'openvino_model/saved_model.xml'
compiled_model = core.compile_model(model=model_path, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
logging.info("OpenVINO model loaded successfully")

# 새로운 비디오에서 이상 행동 감지
logging.info("Starting anomaly detection for new video")
anomaly_scores = predict_anomaly("test_0717.mp4")

# 이상 행동 점수 시각화
if anomaly_scores.size > 0:  # Use .size to check if the array is not empty
    logging.info("Plotting anomaly scores")
    plt.figure(figsize=(10, 4))
    plt.plot(anomaly_scores)
    plt.xlabel('Frame')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Scores Over Time')
    plt.show()
else:
    logging.warning("No anomaly scores to plot")

logging.info("Script execution completed")
