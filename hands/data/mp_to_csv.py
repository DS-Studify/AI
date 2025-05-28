import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# MediaPipe 초기화
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# 데이터 통합 리스트
all_data = []

# 각도 계산용 landmark 인덱스
angle_indices = [
    (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9,10,11), (10,11,12),
    (13,14,15), (14,15,16),
    (17,18,19), (18,19,20)
]

# pen / nopen 폴더 경로 및 라벨 설정
datasets = [
    {"folder": "handsPic/pen", "label": 1},
    {"folder": "handsPic/nopen", "label": 0}
]

# 폴더별 데이터 처리
for dataset in datasets:
    folder = dataset["folder"]
    label = dataset["label"]
    image_files = sorted(os.listdir(folder))

    for img_file in image_files:
        img_path = os.path.join(folder, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"이미지 로드 실패: {img_path}")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                base = coords[0]
                rel_coords = [(x - base[0], y - base[1], z - base[2]) for x, y, z in coords]
                flat_coords = [v for pt in rel_coords for v in pt]

                angles = [calculate_angle(coords[i1], coords[i2], coords[i3]) for i1, i2, i3 in angle_indices]
                row = flat_coords + angles + [label]
                all_data.append(row)

# 컬럼명 지정
xyz_cols = [f"{axis}{i}" for i in range(1, 22) for axis in ['x', 'y', 'z']]
angle_cols = [f"angle{i+1}" for i in range(10)]
columns = xyz_cols + angle_cols + ['label']

# CSV 저장
df = pd.DataFrame(all_data, columns=columns)
df.to_csv("hand_relative.csv", index=False)
print("CSV 파일 저장 완료: hand_relative.csv")
