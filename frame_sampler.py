import os
import cv2

def extract_frames_from_videos(input_folder, output_folder, k):
    """
    Extract the first k frames from each video in the input folder
    and save them to the output folder.

    Args:
        input_folder (str): Path to the folder containing input videos.
        output_folder (str): Path to the folder where frames will be saved.
        k (int): Number of frames to extract from each video.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)
        if not os.path.isfile(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_file}")
            continue

        video_name = os.path.splitext(video_file)[0]
        frame_count = 0

        while frame_count < k:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video reached or error reading video: {video_file}")
                break

            frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count + 1}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()
        print(f"Extracted {frame_count} frames from {video_file}")

# Example usage
input_folder = "/home/hpnquoc/Desktop/suwon/suwon#54/"  # Replace with the path to your video folder
output_folder = "dummy/test"  # Replace with the path to your target folder
k = 1  # Number of frames to extract

extract_frames_from_videos(input_folder, output_folder, k)
