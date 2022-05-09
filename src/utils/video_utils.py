import math
import numpy as np

from cv2 import cv2


def get_num_frames_from_videos(path, num_extraction):
    required_frames_per_sec = 10
    cap = cv2.VideoCapture(path)  # capturing the video from the given path
    frame_rate = cap.get(5)  # frame rate
    if frame_rate not in range(1, 50):  # make it more logical
        frame_rate = 25
    rate_divisor = max(math.ceil(frame_rate / required_frames_per_sec), 1)
    num_frame = cap.get(7)
    video_duration = round(float(num_frame) / float(frame_rate), 2)
    second_divisor = max(math.floor(video_duration / num_extraction), 1)
    total_created_frame = 0
    frames = []
    while cap.isOpened():
        frame_id = cap.get(1)
        ret, frame = cap.read()
        current_second = int(frame_id / frame_rate)
        if current_second % second_divisor == 0:
            if not ret:
                break
            if frame_id % rate_divisor == 0:
                total_created_frame += 1
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        else:
            if not ret:
                break

    cap.release()
    frames_array = np.asarray(frames)
    return frames_array


def get_all_frames_from_videos(path):
    cap = cv2.VideoCapture(path)  # capturing the video from the given path
    total_created_frame = 0
    frames = []
    ret = True
    while cap.isOpened() and ret:
        ret, frame = cap.read()
        if ret:
            total_created_frame += 1
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    cap.release()
    frames_array = np.asarray(frames)
    return frames_array
