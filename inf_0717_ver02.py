import os
import cv2
import numpy as np
from openvino.runtime import Core, Tensor
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
import logging
from tqdm import tqdm

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
        if i >= max_length:
            break
        if frame.shape == (17, 2):
            padded_segment[i] = frame
        else:
            padded_segment[i] = np.zeros((17, 2))
    return padded_segment

# 이상 행동 감지 함수
def predict_anomaly(video_path):
    logging.info(f"Starting anomaly prediction for video: {video_path}")
    max_length = 119

    extract_keypoints_from_video(video_path, "temp_output")
    
    frames = [np.load(os.path.join("temp_output", f)) for f in sorted(os.listdir("temp_output")) if f.endswith(".npy")]
    logging.info(f"Loaded {len(frames)} frames for prediction")

    all_predictions = []
    step_size = 1
    scale_factor = 1e10  # 편차를 크게 만들기 위한 스케일 팩터
    for start in tqdm(range(0, len(frames), step_size), desc="Processing frames"):
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
        anomaly_scores = [pred * scale_factor for pred in predictions.flatten()]  # 특정 수를 곱해서 편차를 크게 만듦
        
        # 예측 값 로그 출력
        logging.info(f"Anomaly scores for segment starting at frame {start}: {anomaly_scores}")

        all_predictions.extend(anomaly_scores)
        logging.info(f"Processed segment. Current total predictions: {len(all_predictions)}")
    
    if len(all_predictions) > 0:
        min_score = np.min(all_predictions)
        max_score = np.max(all_predictions)
        normalized_scores = [(score - min_score) / (max_score - min_score) * 100 for score in all_predictions]
        
        # 스무딩 제거
        smooth_scores = normalized_scores
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
anomaly_scores = predict_anomaly("test_2.mp4")

# 입력 동영상 로드
video_path = "test_2.mp4"
cap = cv2.VideoCapture(video_path)
output_video_path = "result_with_graph_test.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height * 2))  # 높이를 2배로 설정

# Matplotlib 설정
matplotlib.use('Agg')

frame_count = 0
for _ in tqdm(range(total_frames), desc="Generating video with graph"):
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count < len(anomaly_scores):
        score = anomaly_scores[frame_count]

        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        ax.plot(anomaly_scores[:frame_count+1])
        ax.set_xlim([0, total_frames])
        ax.set_ylim([0, 100])
        ax.set_title('Anomaly Scores Over Time')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Anomaly Score')

        fig.canvas.draw()
        graph_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        graph_img = graph_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)
        plt.close(fig)

        graph_img = cv2.resize(graph_img, (width, height))  # 그래프 이미지를 동영상 프레임의 너비와 높이로 맞춤
        combined_frame = np.vstack((frame, graph_img))

        out.write(combined_frame)

    frame_count += 1

cap.release()
out.release()
logging.info("Result video with graph saved successfully")
