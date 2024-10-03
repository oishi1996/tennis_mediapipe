import cv2
import numpy as np
import mediapipe as mp

# 色を16進数で指定し、BGR形式に変換
background_color = (222, 196, 176)  # #b0c4de
skeleton_point_color = (160, 158, 95)  # #5f9ea0
skeleton_line_color = (139, 139, 0)  # #e6e6fa

#プロットサイズ
skeleton_point_size = 15
skeleton_line_size = 5

# MediaPipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 動画の読み込み
video_path = 'data/so_serve_2.mp4'
cap = cv2.VideoCapture(video_path)

# 出力動画の設定
output_path = 'output_video_2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

# MediaPipe Poseの設定
with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 骨格推定処理
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 骨格推定動画を描画
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 骨格情報の可視化 (右側の空の画像)
        skeleton_image = np.full((height, width, 3), background_color, dtype=np.uint8)  # 背景色を設定
        if results.pose_landmarks:
            # 骨格ポイントを描画
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(skeleton_image, (x, y), skeleton_point_size, skeleton_point_color, -1)

            # 骨格ラインを描画
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start_point = results.pose_landmarks.landmark[start_idx]
                end_point = results.pose_landmarks.landmark[end_idx]
                start_x = int(start_point.x * width)
                start_y = int(start_point.y * height)
                end_x = int(end_point.x * width)
                end_y = int(end_point.y * height)
                cv2.line(skeleton_image, (start_x, start_y), (end_x, end_y), skeleton_line_color, skeleton_line_size)

        # 2つの画像を結合
        combined_frame = np.hstack((frame, skeleton_image))

        # 出力動画に書き込み
        out.write(combined_frame)

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()
