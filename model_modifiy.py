import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model

# 기존 모델 로드
model = tf.keras.models.load_model('anomaly_detection_model.h5')

# 새로운 입력 레이어 정의
input_shape = (119, 34)
new_input = Input(shape=input_shape, name='lstm_input')

# 기존 모델의 레이어를 새로운 입력 레이어에 연결
x = LSTM(64, return_sequences=True)(new_input)
x = LSTM(64)(x)
output = Dense(1)(x)

# 새로운 모델 생성
new_model = Model(inputs=new_input, outputs=output)

# 새로운 모델 저장
new_model.save('anomaly_detection_model_modified.h5')
tf.saved_model.save(new_model, 'saved_model_dir')
