import subprocess

def frame_extraction(video_path, output_path, frame_rate=1, name='frame'):
    cmd = f'ffmpeg -i "{video_path}" -r {frame_rate} "{output_path}/{name}_%d.png"'
    returned_value = subprocess.call(cmd, shell=True)
    return returned_value

if __name__ == '__main__':
    r = frame_extraction('dummy/video/dramatic_kitten_meme.mp4', 'output/test', 60, 'frame')
    print(r)
    print('Frames extracted successfully!')