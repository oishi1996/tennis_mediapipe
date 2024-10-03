import cv2
import mediapipe as mp

# Mediapipeの設定
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# 動画のパスを指定
video_path = 'data/so_serve_1.mp4'
output_path = 'output/skeleton_output.mp4'

# 動画の読み込み
cap = cv2.VideoCapture(video_path)

# フレームサイズを取得
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 出力動画の設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("フレームの取得に失敗しました。動画の終わりに達しました。")
        break
    
    # 骨格推定を行う
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)
    
    # 骨格の描画
    if result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # 骨格が描画されたフレームを出力動画に書き込む
    out.write(frame)
    
    # フレームを表示
    cv2.imshow("Skeleton", frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()
