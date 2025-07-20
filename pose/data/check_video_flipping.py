import cv2
import os
from glob import glob

# 체크할 폴더
folders = [
    'data/good_pose',
    'data/nfocus_lean_back',
    'data/nfocus_lean_forward',
    'data/nfocus_lean_side',
    'data/sleep_head_back',
    'data/sleep_head_down'
]

# 각 폴더 내 모든 mp4 영상 수집
video_paths = []
for folder in folders:
    video_paths.extend(glob(os.path.join(folder, '*.mp4')))

# 각 영상 순회 확인
for video_path in sorted(video_paths):
    cap = cv2.VideoCapture(video_path)
    filename = os.path.basename(video_path)

    print(f"확인 중: {filename}")
    frame_idx = 0
    show_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 15프레임마다 1장씩, 최대 5장만 확인
        if frame_idx % 15 == 0 and show_count < 5:
            frame_show = cv2.resize(frame, (640, 360))
            cv2.putText(frame_show, f'FLIP CHECK: {filename}', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Flip Check", frame_show)
            show_count += 1

            # 1초 기다림 or 바로 넘기기
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

print("모든 영상 확인 완료")
