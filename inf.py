import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import matplotlib.pyplot as plt

# 키포인트 추출 함수
def extract_keypoints_from_video(video_path, output_folder):
    model = YOLO('yolov8n-pose.pt')
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)  # YOLOv8을 사용하여 포즈 예측
        
        # keypoints 확인 및 단순화
        keypoints = results[0].keypoints
        if keypoints is not None:
            keypoints = keypoints.xy.cpu().numpy()  # xy 좌표를 numpy 배열로 변환
            # keypoints의 구조 확인
            print(f"Frame {frame_count}: keypoints shape {keypoints.shape}")

            # keypoints를 저장 가능한 형태로 변환 (예: 2D 배열로 변환)
            keypoints = keypoints.reshape(-1, keypoints.shape[-1])
            np.save(os.path.join(output_folder, f"{os.path.basename(video_path).split('.')[0]}_frame_{frame_count:05d}.npy"), keypoints)
        
        frame_count += 1
    cap.release()

# 이상 행동 감지 함수
def predict_anomaly(video_path):
    # 저장된 max_length 로드
    max_length = np.load('max_length.npy')
    
    extract_keypoints_from_video(video_path, "temp_output")
    
    frames = [np.load(os.path.join("temp_output", f)) for f in sorted(os.listdir("temp_output")) if f.endswith(".npy")]
    
    padded_frames = []
    for frame in frames:
        if frame.shape[0] < max_length:
            padding = np.zeros((max_length - frame.shape[0], frame.shape[1]))
            frame = np.vstack((frame, padding))
        padded_frames.append(frame)
    
    padded_frames = np.array(padded_frames)
    padded_frames = np.nan_to_num(padded_frames)
    
    predictions = model.predict(padded_frames)
    anomaly_scores = [pred[0] for pred in predictions]
    return anomaly_scores

# 모델 불러오기
model = load_model('anomaly_detection_model.h5')

# 새로운 비디오에서 이상 행동 감지
anomaly_scores = predict_anomaly("test.mp4")

# 이상 행동 점수 시각화
plt.figure(figsize=(10, 4))
plt.plot(anomaly_scores)
plt.xlabel('Frame')
plt.ylabel('Anomaly Score')
plt.title('Anomaly Scores Over Time')
plt.show()
