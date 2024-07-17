# import tensorflow as tf

# # 모델 로드
# model = tf.keras.models.load_model('anomaly_detection_model.h5')

# # 모델 요약 정보 출력 (입력 형태 포함)
# model.summary()

# # 또는 입력 형태만 출력
# print(model.input_shape)

import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model('anomaly_detection_model.h5')

# 모델 입력 이름 출력
print([input.name for input in model.inputs])
