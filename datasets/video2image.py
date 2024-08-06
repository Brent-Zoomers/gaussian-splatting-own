import cv2
import os

def video_to_frames(video_path, output_folder):
    # Create the output directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as JPEG file
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Video has been successfully converted to {frame_count} frames.")

if __name__ == "__main__":
    video_path = 'VID_20240804_191509.mp4'  # Replace with your video file path
    output_folder = 'parents'  # Replace with your desired output folder
    video_to_frames(video_path, output_folder)
