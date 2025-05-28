import cv2
import os

def extract_frames(video_path, output_folder):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return

    # FPS 정보 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    # 0.5초마다 저장할 프레임 간격 계산
    frame_interval = int(fps * 0.5)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            flipped_frame = cv2.flip(frame, 0)  # 사진 수직으로 뒤집기
            frame_filename = os.path.join(output_folder, f"frame_no_left_mk_{saved_count:05d}.jpg")  # 저장될 파일명
            cv2.imwrite(frame_filename, flipped_frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"총 {saved_count}개의 프레임을 저장했습니다 (0.5초마다).")


# 경로
video_path = "handsPic/video/nopenLeftMK.mp4"  # 비디오 경로
output_folder = "handsPic/nopen"  # 저장될 폴더
extract_frames(video_path, output_folder)
