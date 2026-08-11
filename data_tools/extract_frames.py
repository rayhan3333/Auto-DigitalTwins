import os
import cv2
from glob import glob
from argparse import ArgumentParser


def extract_frames(video_path, output_dir, interval=1, resize=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    image_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        if resize > 1:
            frame = cv2.resize(frame, (width//resize, height//resize))
        if image_count > 100:
            if frame_count % (interval * 1) == 0:
                cv2.imwrite(os.path.join(output_dir, f'{image_count:06d}.png'), frame)
                image_count += 1
        else:
            if frame_count % interval == 0:
                cv2.imwrite(os.path.join(output_dir, f'{image_count:06d}.png'), frame)
                image_count += 1
        frame_count += 1
    cap.release()
    print(f'Extracted {image_count} frames from {video_path}, interval: {interval}, resize: {resize}')
    return image_count


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="/mnt/fillipo/yuliu/VideoArtGS/videos", help="path to the video directory")
    parser.add_argument("--data_dir", type=str, default="/mnt/fillipo/yuliu/VideoArtGS/data/artgs/realscan", help="path to the scene directory")
    parser.add_argument("--video_name", type=str, default="storage_1r", help="name of the video")
    parser.add_argument("--interval", type=int, default=2, help="interval between frames")
    parser.add_argument("--resize", type=int, default=2, help="resize factor")
    args = parser.parse_args()

    video_dir = args.video_dir
    data_dir = args.data_dir
    video_name = args.video_name
    video_path = os.path.join(video_dir, f'{video_name}.mp4')
    scene_path = os.path.join(data_dir, video_name)
    image_dir = os.path.join(scene_path, "images")
    os.makedirs(image_dir, exist_ok=True)
    os.system(f"rm {image_dir}/*")
    extract_frames(video_path, image_dir, interval=args.interval, resize=args.resize)
