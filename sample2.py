import cv2

video_path = 'data/so_serve_1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("動画の読み込みに失敗しました。")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"フレームレート: {fps} fps")
print(f"総フレーム数: {total_frames} フレーム")

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("フレームの取得に失敗しました。動画の終わりに達しました。")
        break
    
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # 現在のフレーム位置を取得
    print(f"取得したフレーム: {current_frame}")  # デバッグ用メッセージ

    # フレームに対する処理（例: 骨格推定）
    cv2.imshow("Frame", frame)

    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
