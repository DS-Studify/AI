import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from keras.models import load_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import joblib

# 파일 경로 설정
CSV_PATH = '../filtered_pose_face_data.csv'
MODEL_PATH = 'mlp_pose.h5'
SCALER_PATH = 'scaler.pkl'
ENCODER_PATH = 'label_encoder.pkl'

# 데이터 로드
df = pd.read_csv(CSV_PATH)
X = df.drop(columns=['label']).values
y_true = df['label'].values

# 전처리
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)

X_scaled = scaler.transform(X)
y_encoded = label_encoder.transform(y_true)
y_categorical = to_categorical(y_encoded)

# 모델 로드
model = load_model(MODEL_PATH)

# 예측
y_pred = np.argmax(model.predict(X_scaled, verbose=0), axis=1)

# 정확도
correct = np.sum(y_pred == y_encoded)
total = len(y_encoded)
acc = correct / total
print(f"Exact Accuracy: {acc:.4f} ({correct} / {total})")

# prefix 기준 정확도
def match_prefix(true_label, pred_label, prefix_words=2):
    true_prefix = '_'.join(true_label.split('_')[:prefix_words])
    pred_prefix = '_'.join(pred_label.split('_')[:prefix_words])
    return true_prefix == pred_prefix

y_pred_label = label_encoder.inverse_transform(y_pred)
y_true_label = label_encoder.inverse_transform(y_encoded)

correct_prefix = sum(match_prefix(t, p) for t, p in zip(y_true_label, y_pred_label))
prefix_acc = correct_prefix / total
print(f"Approx Accuracy (prefix match): {prefix_acc:.4f} ({correct_prefix} / {total})")

# 혼동행렬 시각화
cm = confusion_matrix(y_encoded, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix: mlp_pose")
plt.tight_layout()
plt.show()
