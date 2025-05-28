import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import joblib

# 1. CSV 불러오기
df = pd.read_csv('../hand_relative_augmented.csv')

# 2. 입력(X), 출력(y) 분리
xyz_cols = [col for col in df.columns if col.startswith(('x', 'y', 'z'))]
angle_cols = [col for col in df.columns if col.startswith('angle')]
X_coord = df[xyz_cols].values
X_angle = df[angle_cols].values
X = np.concatenate([X_coord, X_angle], axis=1)
y = df['label'].values

# 3. 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. 모델 구성
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 6. 컴파일 및 학습
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

# 7. 평가 및 저장
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

model.save('pen_detection_mlp_relative.h5')
joblib.dump(scaler, 'scaler_relative.save')
