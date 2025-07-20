import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import mediapipe as mp

# 경로 설정
model_path = 'mlp_pose.h5'
scaler_path = 'scaler.pkl'
encoder_path = 'label_encoder.pkl'
video_path = 'study_2.mp4'
output_path = 'mlp_pose_result.mp4'
input_dim = 93

# 모델 및 도구 로드
model = load_model(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

# Mediapipe 초기화
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose()
face = mp_face.FaceMesh(refine_landmarks=True)

# 영상 설정
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = max(1, int(fps // 3))
output_size = (640, 360)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

def extract_landmark_vector(face_result, pose_result):
    face_idx = [1, 4, 10, 13, 14, 33, 78, 133, 145, 152,
                159, 199, 234, 263, 308, 362, 374, 386, 454]
    pose_idx = list(range(11, 23))

    landmarks = []

    # 얼굴 좌표 추출 (19개)
    if face_result.multi_face_landmarks:
        for idx in face_idx:
            lm = face_result.multi_face_landmarks[0].landmark[idx]
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * len(face_idx) * 3)

    # 포즈 좌표 추출 (12개)
    if pose_result.pose_landmarks:
        for idx in pose_idx:
            lm = pose_result.pose_landmarks.landmark[idx]
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * len(pose_idx) * 3)

    return landmarks

# 프레임 처리
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % interval != 0:
        frame_idx += 1
        continue

    frame = cv2.resize(frame, output_size)
    # frame = cv2.flip(frame, 0)  # 상하 반전
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    pose_result = pose.process(image)
    face_result = face.process(image)
    row = extract_landmark_vector(face_result, pose_result)

    if len(row) == input_dim:
        X_input = scaler.transform([row])
        y_pred = model.predict(X_input, verbose=0)
        label = label_encoder.inverse_transform([np.argmax(y_pred)])[0]
        cv2.putText(frame, f'{label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Pose Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"영상 처리 완료, 저장 위치: {output_path}")
