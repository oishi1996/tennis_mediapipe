from flask import Flask, request, jsonify, url_for
import os
import cv2
import numpy as np
import mediapipe as mp


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 動画の処理関数
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    background_color = (222, 196, 176)
    skeleton_point_color = (160, 158, 95)
    skeleton_line_color = (139, 139, 0)

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

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

            combined_frame = np.hstack((frame, skeleton_image))
            out.write(combined_frame)

    cap.release()
    out.release()

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output_video.mp4')
        process_video(file_path, output_path)

        return jsonify({'video_url': url_for('static', filename='output/output_video.mp4')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
