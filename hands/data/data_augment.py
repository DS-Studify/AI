import pandas as pd
import numpy as np

# CSV 파일 불러오기
df = pd.read_csv('hand_relative.csv')

# 데이터 분리
labels = df['label'].values
xyz_cols = [col for col in df.columns if col.startswith(('x', 'y', 'z'))]
angle_cols = [col for col in df.columns if col.startswith('angle')]

landmarks = df[xyz_cols].values  # 63개 (상대좌표)
angles = df[angle_cols].values   # 10개 (각도값)

# ===== 증강 함수들 =====
# 각도값 보존 / 좌표값만 증강

# 노이즈 추가 (z축은 약하게)
def add_noise(data, noise_level=0.01, z_noise_level=0.005):
    noise = np.random.normal(0, noise_level, data.shape)
    for i in range(2, data.shape[1], 3):  # z좌표 별도 처리
        noise[:, i] = np.random.normal(0, z_noise_level, size=data.shape[0])
    return data + noise

# 스케일 증강
def scale_data(data, scale_range=(0.7, 1.3)):
    scale = np.random.uniform(*scale_range)
    return data * scale

# 회전 증강 (x, y만)
def rotate_data(data, angle_range=(-30, 30)):
    angle = np.radians(np.random.uniform(*angle_range))
    cos_val, sin_val = np.cos(angle), np.sin(angle)
    rotated = data.copy()

    for i in range(0, data.shape[1] - 2, 3):
        x = data[:, i]
        y = data[:, i + 1]
        x_rot = x * cos_val - y * sin_val
        y_rot = x * sin_val + y * cos_val
        rotated[:, i] = x_rot
        rotated[:, i + 1] = y_rot

    return rotated

# 평행이동 증강 (x, y만 이동)
def translate_data(data, translate_range=(-0.05, 0.05)):
    tx = np.random.uniform(*translate_range)
    ty = np.random.uniform(*translate_range)
    data[:, ::3] += tx  # x좌표 (0,3,6,9,...)
    data[:, 1::3] += ty  # y좌표 (1,4,7,10,...)
    return data

# ===== 증강 데이터 생성 =====

augmented_data = []
augmented_labels = []

for i in range(len(landmarks)):
    original = landmarks[i].reshape(1, -1)
    original_angle = angles[i].reshape(1, -1)
    label = labels[i]

    # 원본 + 증강 4종
    for aug_func in [lambda x: x, add_noise, scale_data, rotate_data, translate_data]:
        new_landmark = aug_func(original.copy())
        combined = np.concatenate([new_landmark, original_angle], axis=1)  # 각도는 그대로
        augmented_data.append(combined)
        augmented_labels.append(label)

# ===== 결과 저장 =====

augmented_data = np.vstack(augmented_data)
augmented_labels = np.array(augmented_labels)

# 최종 데이터프레임 생성
augmented_df = pd.DataFrame(augmented_data, columns=xyz_cols + angle_cols)
augmented_df['label'] = augmented_labels

# CSV로 저장
augmented_df.to_csv('hand_relative_augmented.csv', index=False)
print("데이터 증강 완료: hand_relative_augmented.csv")
