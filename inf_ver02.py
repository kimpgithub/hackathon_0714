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

# 이동 평균 함수
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

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
    
    # 이상 행동 점수 정규화 (0에서 100 사이)
    min_score = np.min(anomaly_scores)
    max_score = np.max(anomaly_scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) * 100 for score in anomaly_scores]
    
    # 이동 평균을 사용하여 점수 부드럽게 만들기
    smooth_scores = moving_average(normalized_scores, window_size=5)
    
    return smooth_scores

# 모델 불러오기
model = load_model('anomaly_detection_model.h5')

# 새로운 비디오에서 이상 행동 감지
anomaly_scores = predict_anomaly("test.mp4")

# 비디오 캡처 설정
video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 비디오 작성을 위한 설정
output_video_path = 'output_with_graph.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height + 200))  # 그래프를 위해 높이를 200 늘림

# 그래프를 그릴 수 있는 설정
fig, ax = plt.subplots(figsize=(frame_width / 100, 2))  # DPI를 고려하여 크기 설정
plt.close(fig)

for i, score in enumerate(anomaly_scores):
    ret, frame = cap.read()
    if not ret:
        break

    # 비디오 프레임 업데이트
    if i < len(anomaly_scores):
        ax.clear()
        ax.plot(anomaly_scores[:i + 1])
        ax.set_xlim([0, len(anomaly_scores)])
        ax.set_ylim([0, 100])
        ax.set_xlabel('Frame')
        ax.set_ylabel('Anomaly Score')
        ax.set_title('Anomaly Scores Over Time')

        # 그래프 이미지를 PNG로 저장
        fig.canvas.draw()
        graph_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)

        # 그래프 이미지를 비디오 프레임 아래에 붙이기
        combined_frame = np.vstack((frame, graph_img))

        # 비디오에 프레임 쓰기
        out.write(combined_frame)

cap.release()
out.release()

print("Video with graph has been saved successfully.")
