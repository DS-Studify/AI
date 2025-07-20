import cv2
import mediapipe as mp
import csv
import os
from glob import glob

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

# 상체 랜드마크 (0~22번)
upper_body_idx = list(range(23))

# 주요 FaceMesh 인덱스 (총 19개)
face_idx = sorted([1, 4, 10, 13, 14, 33, 78, 133, 145, 152,
                   159, 199, 234, 263, 308, 362, 374, 386, 454])

# 헤더 구성
header = []
for idx in face_idx:
    header += [f'fx{idx}', f'fy{idx}', f'fz{idx}']
for i in upper_body_idx:
    header += [f'x{i}', f'y{i}', f'z{i}']
header.append('label')

# 처리 대상 폴더 및 라벨
target_folders = {
    'data/good_pose': 'good_pose',
    'data/nfocus_lean_back': 'nfocus_lean_back',
    'data/nfocus_lean_forward': 'nfocus_lean_forward',
    'data/nfocus_lean_side': 'nfocus_lean_side',
    'data/sleep_head_back': 'sleep_head_back',
    'data/sleep_head_down': 'sleep_head_down'
}

# 저장할 CSV 경로
output_csv = 'filtered_pose_face_data.csv'
results_data = []

for folder_path, label in target_folders.items():
    video_paths = glob(os.path.join(folder_path, '*.mp4'))
    print(f'\n처리 중: {folder_path} ({len(video_paths)}개 파일)')

    for video_path in sorted(video_paths):
        filename = os.path.basename(video_path)
        flip_flag = "flip" in filename

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        interval = max(1, int(fps // 3))

        with mp_pose.Pose(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose, \
             mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                              refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % interval != 0:
                    frame_idx += 1
                    continue

                if flip_flag:
                    frame = cv2.flip(frame, 0)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                pose_result = pose.process(image)
                face_result = face_mesh.process(image)

                row = []

                # 얼굴 좌표
                if face_result.multi_face_landmarks:
                    face = face_result.multi_face_landmarks[0]
                    for idx in face_idx:
                        lm = face.landmark[idx]
                        row.extend([lm.x, lm.y, lm.z])
                else:
                    row.extend([0.0] * (len(face_idx) * 3))

                # 포즈 좌표 (v 제외)
                if pose_result.pose_landmarks:
                    for i in upper_body_idx:
                        lm = pose_result.pose_landmarks.landmark[i]
                        row.extend([lm.x, lm.y, lm.z])
                else:
                    row.extend([0.0] * (len(upper_body_idx) * 3))

                row.append(label)
                results_data.append(row)
                frame_idx += 1

        cap.release()

# 0값 5개 이상 포함된 행 제거
# filtered_data = [row for row in results_data if row[:-1].count(0.0) < 5]
# 포즈 기준 0.0 개수 판단
filtered_data = []
for row in results_data:
    pose_only = row[len(face_idx)*3:-1]  # 얼굴 좌표 이후 ~ 라벨 직전
    if pose_only.count(0.0) < 5:
        filtered_data.append(row)

# CSV 저장
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(filtered_data)

# 로그 출력
print(f"총 추출된 데이터 수: {len(results_data)}")
print(f"유효한 데이터 수: {len(filtered_data)}")
print(f"저장 파일: {output_csv}")
