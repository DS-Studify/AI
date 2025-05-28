import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib

# === 1. 모델과 스케일러 로드 ===
model = tf.keras.models.load_model('pen_detection_mlp_relative.h5')
scaler = joblib.load('scaler_relative.save')

# === 2. MediaPipe 손 추적기 초기화 ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === 3. 각도 계산 함수 ===
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

# === 4. 각도 계산 인덱스 ===
angle_indices = [
    (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9,10,11), (10,11,12),
    (13,14,15), (14,15,16),
    (17,18,19), (18,19,20)
]

# === 5. 동영상 파일 열기 ===
video_path = "study_2.mp4"
cap = cv2.VideoCapture(video_path)

# === 6. 결과 저장 영상 설정 ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('study_output_2.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# === 7. 해상도 축소 비율 (ex: 0.5배 축소) ===
scale_ratio = 0.5
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 0)
    frame_count += 1

    # 2초 간격 인식 (30fps 기준 → 60프레임마다 1번 처리)
    if frame_count % 30 != 0:
        out.write(frame)  # 그대로 저장
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # 해상도 축소 (MediaPipe 처리용)
    resized_frame = cv2.resize(frame, (int(frame.shape[1] * scale_ratio), int(frame.shape[0] * scale_ratio)))

    # === MediaPipe 손 인식 ===
    image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    label_text = "No Pen"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            # 손목 기준 상대좌표로 변환
            base = coords[0]
            rel_coords = [(x - base[0], y - base[1], z - base[2]) for x, y, z in coords]
            flat_coords = [v for pt in rel_coords for v in pt]  # 63차원

            # 관절 각도값 계산 (10개)
            angles = [calculate_angle(coords[i1], coords[i2], coords[i3]) for i1, i2, i3 in angle_indices]

            # 최종 feature 벡터 구성
            input_vector = np.array(flat_coords + angles).reshape(1, -1)
            input_scaled = scaler.transform(input_vector)

            # 예측
            prediction = model.predict(input_scaled)
            label = 1 if prediction[0][0] > 0.5 else 0

            if label == 1:
                label_text = "Pen in Hand"
                break  # 하나라도 펜이면 해당 프레임은 pen

    # 텍스트 시각화
    color = (0, 255, 0) if label_text == "Pen in Hand" else (0, 0, 255)
    cv2.putText(frame, label_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 랜드마크 시각화
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 영상 저장 및 출력
    out.write(frame)
    cv2.imshow('Video Test - Pen Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
