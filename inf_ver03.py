from flask import Flask, request, jsonify, send_file, render_template_string
import os
import cv2
import numpy as np
from ultralytics import YOLO
from openvino.runtime import Core

app = Flask(__name__)

# 업로드 디렉토리 생성
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the OpenVINO model
core = Core()
model_path = 'openvino_model/saved_model.xml'
compiled_model = core.compile_model(model=model_path, device_name="CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# HTML 템플릿
upload_page = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Video Anomaly Detection</title>
  </head>
  <body>
    <div class="container">
      <h1 class="mt-5">Upload Video for Anomaly Detection</h1>
      <form action="/predict_anomaly" method="post" enctype="multipart/form-data">
        <div class="form-group">
          <label for="file">Choose video file</label>
          <input type="file" class="form-control" id="file" name="file" required>
        </div>
        <button type="submit" class="btn btn-primary">Upload</button>
      </form>
    </div>
  </body>
</html>
'''

# 키포인트 추출 함수
def extract_keypoints_from_video(video_path, output_folder):
    yolo_model = YOLO('yolov8n-pose.pt')
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = yolo_model(frame)  # YOLOv8을 사용하여 포즈 예측
        
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
    
    # OpenVINO 모델을 사용한 추론 수행
    predictions = compiled_model([padded_frames])[output_layer]
    anomaly_scores = [pred[0] for pred in predictions]
    
    # 이상 행동 점수 정규화 (0에서 100 사이)
    min_score = np.min(anomaly_scores)
    max_score = np.max(anomaly_scores)
    normalized_scores = [(score - min_score) / (max_score - min_score) * 100 for score in anomaly_scores]
    
    # 결과를 파일로 저장
    result_file = "results.txt"
    np.savetxt(result_file, normalized_scores, fmt='%.2f')
    
    return result_file

@app.route('/')
def upload_file():
    return render_template_string(upload_page)

@app.route('/predict_anomaly', methods=['POST'])
def predict_anomaly_route():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        result_file = predict_anomaly(file_path)
        return send_file(result_file, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
