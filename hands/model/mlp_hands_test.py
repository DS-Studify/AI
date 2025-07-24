import os
import cv2
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.models import load_model

# 1. 모델 및 스케일러 로드
model = load_model('pen_mlp.h5')
scaler = joblib.load('scaler_mlp.save')

# 2. 테스트 데이터셋 불러오기
df = pd.read_csv('hand_test.csv')

# 3. 특성 및 레이블 분리
xyz_cols = [col for col in df.columns if col.startswith(('x', 'y', 'z'))]
angle_cols = [col for col in df.columns if col.startswith('angle')]
X = np.concatenate([df[xyz_cols].values, df[angle_cols].values], axis=1)
y = df['label'].values

# 4. 스케일링
X_scaled = scaler.transform(X)

# 5. 예측
threshold = 0.3
y_probs = model.predict(X_scaled).flatten()
y_pred = (y_probs >= threshold).astype(int)

# 6. 평가
acc = accuracy_score(y, y_pred)
print(f"✅ 정확도 (Accuracy): {acc:.4f}")
print("\n📋 분류 리포트:")
print(classification_report(y, y_pred, target_names=["Not Pen", "Pen"]))

# 7. 혼동 행렬 시각화
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Pen", "Pen"], yticklabels=["Not Pen", "Pen"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Pen Detection")
plt.tight_layout()
plt.show()

# 8. 오분류 이미지 저장
os.makedirs("misclassified", exist_ok=True)

misclassified = df[y != y_pred]  # 오분류된 행 필터링

for idx, row in misclassified.iterrows():
    img_path = row['img_path']
    true_label = row['label']
    pred_label = y_pred[idx]

    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 이미지 로드 실패: {img_path}")
            continue

        # 저장 파일명: 실제_예측_파일명.jpg
        basename = os.path.basename(img_path)
        save_name = f"{true_label}_{pred_label}_{basename}"
        save_path = os.path.join("misclassified", save_name)

        cv2.imwrite(save_path, img)

    except Exception as e:
        print(f"❌ 저장 실패: {img_path} - {e}")