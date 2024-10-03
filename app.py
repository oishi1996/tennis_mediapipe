import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import numpy as np
import mediapipe as mp

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'

# MediaPipeの初期化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 動画の処理関数
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    background_color = (222, 196, 176)  # #b0c4de
    skeleton_point_color = (160, 158, 95)  # #5f9ea0
    skeleton_line_color = (139, 139, 0)  # #e6e6fa

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
            skeleton_image = np.full((height, width, 3), background_color, dtype=np.uint8)
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(skeleton_image, (x, y), 15, skeleton_point_color, -1)

                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_point = results.pose_landmarks.landmark[start_idx]
                    end_point = results.pose_landmarks.landmark[end_idx]
                    start_x = int(start_point.x * width)
                    start_y = int(start_point.y * height)
                    end_x = int(end_point.x * width)
                    end_y = int(end_point.y * height)
                    cv2.line(skeleton_image, (start_x, start_y), (end_x, end_y), skeleton_line_color, 5)

            # 2つの画像を結合
            combined_frame = np.hstack((frame, skeleton_image))
            out.write(combined_frame)

    cap.release()
    out.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # ファイルのアップロード
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # 出力動画のパス
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_video.mp4')

            # 動画の処理を行う
            process_video(file_path, output_path)

            return redirect(url_for('output_video'))

    return render_template('index.html')

@app.route('/output')
def output_video():
    return render_template('output.html', video_url=url_for('static', filename='output/output_video.mp4'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
