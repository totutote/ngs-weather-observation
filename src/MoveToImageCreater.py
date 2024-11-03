import os
import cv2

base_dir = './content/original_video/'
output_dir = './content/dataimagegenerator_input/'
max_height = 256  # 変換後画像の高さ
frame_interval_sec = 0.1  # フレームを抽出する間隔（秒）

# Get the list of label directories
label_dirs = [label for label in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, label))]

# Iterate over each label directory
for label in label_dirs:
    label_dir = os.path.join(base_dir, label)
    output_label_dir = os.path.join(output_dir, label)
    os.makedirs(output_label_dir, exist_ok=True)

    # Get the list of video files in the label directory
    video_files = [file for file in os.listdir(label_dir) if file.endswith('.mp4')]

    # Iterate over each video file
    for video_file in video_files:
        video_path = os.path.join(label_dir, video_file)

        # Read the video file
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # フレームレートを取得
        frame_interval = int(fps * frame_interval_sec)  # フレーム間隔をフレーム数に変換

        # Iterate over each frame in the video
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Check if the current frame is at the specified interval
            if frame_count % frame_interval == 0:
                # Resize the frame to have a maximum height of max_height pixels while maintaining the aspect ratio
                height, width = frame.shape[:2]
                if height > max_height:
                    new_height = max_height
                    new_width = int((new_height / height) * width)
                    frame = cv2.resize(frame, (new_width, new_height))

                # Convert the frame to JPG format
                output_file = os.path.join(output_label_dir, f'{video_file}_{frame_count}.jpg')
                cv2.imwrite(output_file, frame)

            frame_count += 1

        cap.release()

print('Video files converted to JPG format and resized to a maximum height of', max_height, 'pixels at intervals of', frame_interval_sec, 'seconds.')