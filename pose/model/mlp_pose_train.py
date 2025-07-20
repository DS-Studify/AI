import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.utils import to_categorical
import joblib

# 파일 경로 설정
CSV_PATH = '../filtered_pose_face_data.csv'
MODEL_PATH = 'mlp_pose.h5'
SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'label_encoder.pkl'

# 데이터 로드
df = pd.read_csv(CSV_PATH)
X = df.drop(columns=['label']).values
y = df['label'].values

# 전처리
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 모델 정의
model = Sequential([
    Dense(128, input_shape=(X.shape[1],)),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(32),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.3),

    Dense(len(label_encoder.classes_), activation='softmax')
])

# 학습
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_scaled, y_categorical, epochs=30, batch_size=64, verbose=1)

# 저장
model.save(MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(label_encoder, ENCODER_PATH)

print("모델 학습 및 저장 완료")
