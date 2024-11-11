import os
import argparse
import subprocess

argparser = argparse.ArgumentParser()
argparser.add_argument('--video_path', help='Path to the video file.')
argparser.add_argument('--output_path', help='Path to the output directory.')
argparser.add_argument('--frame_rate', help='Frame rate for the extraction.', default=1)
argparser.add_argument('--name', help='Name of the extracted frames.', default='frame')
argparser.add_argument('--task', type=str, default='frame_extraction', help='Task to perform.')
argparser.add_argument('--ext', help='Extension of the extracted frames.', default='png')
args = argparser.parse_args()


def frame_extraction(video_path, output_path, frame_rate=1, name='frame', ext='png'):
    os.makedirs(output_path, exist_ok=True)
    cmd = f'ffmpeg -i "{video_path}" -r {frame_rate} "{output_path}/{name}_%d.{ext}"'
    returned_value = subprocess.call(cmd, shell=True)
    return returned_value

if __name__ == '__main__':
    if args.task == 'frame_extraction':
        r = frame_extraction(args.video_path, args.output_path, args.frame_rate, args.name)
        print('Frames extracted successfully!')