import multiprocessing as mp
import queue, time
import cv2, io, traceback, logging, os
from flask import Flask, render_template, Response
import numpy as np

import logging

VIDEO_SCREEN_SIZE = (640, 480)

from . import functions as f
from .ellipse import fit

WIDTH_CIRCLE = 0.60 # meter
DIAGONAL_FOV_ANGLE_X = 78 # degree
DIAGONAL_FOV_ANGLE_y = DIAGONAL_FOV_ANGLE_X


def detect(queue_s2d, queue_d2s):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 80

    detector = cv2.SimpleBlobDetector_create(params)
    
    while True:
        frame = queue_s2d.get()
        keypoints = detector.detect(frame)
    
        number_of_blobs = len(keypoints)
        location_blobs = np.array([[keypoint.pt[0], keypoint.pt[1]] for keypoint in keypoints])

        if location_blobs.size > 0:

            for i in range(number_of_blobs):
                frame = cv2.circle(frame, (int(location_blobs[i,0]), int(location_blobs[i,1])), 1, (0, 0, 255), 2)
        
            # try:  
            center, width, height, phi = fit(location_blobs)

            frame = cv2.circle(frame, (int(center[0]), int(center[1])), 1, (0, 255, 0), 5)
            frame = cv2.ellipse(
                frame,
                (int(center[0]), int(center[1])),
                (int(width), int(height)),
                phi * 180 / np.pi,
                0,
                360,
                (0, 255, 0),
                2,
            )

            half_width = max(VIDEO_SCREEN_SIZE[0] / (2 * width), VIDEO_SCREEN_SIZE[1] / (2 * height)) * WIDTH_CIRCLE / 2
            height = half_width / np.tan(np.pi / 180 / 2 * DIAGONAL_FOV_ANGLE_X)

            frame = cv2.putText(frame, f"Height: {int(100 * height)}cm", (10,30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0),thickness=1)

            # except Exception:
            #     pass

        frame = cv2.drawKeypoints(
            frame,
            keypoints,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        
        queue_d2s.put(frame)


def stream(queue_s2d, queue_d2s):
    app = Flask(__name__)

    class VideoCapture(cv2.VideoCapture):
        def __init__(self, *args, **kwargs):
            super(VideoCapture, self).__init__(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args, **kwargs):
            self.release()

    @app.route("/")
    def index():
        """Video streaming home page."""
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        """Video streaming route. Put this in the src attribute of an img tag."""
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def gen():
        """Video streaming generator function."""

        frame_mask = np.zeros((VIDEO_SCREEN_SIZE[1], VIDEO_SCREEN_SIZE[0], 3))

        while True:
            _, frame = vc.read()
            frame = cv2.resize(frame, VIDEO_SCREEN_SIZE)

            if queue_s2d.empty():
                queue_s2d.put(frame)

            if not queue_d2s.empty():
                frame_mask = queue_d2s.get()
                
            frame_bool_mask = frame_mask > 0
            frame[frame_bool_mask] = frame_mask[frame_bool_mask]

            _, image_buffer = cv2.imencode(".jpg", frame)
            io_buf = io.BytesIO(image_buffer)

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + io_buf.read() + b"\r\n"
            )

    with VideoCapture(0) as vc:
        app.run(host="0.0.0.0", threaded=True)


if __name__ == "__main__":
    queue_s2d = mp.Queue()
    queue_d2s = mp.Queue()

    process_stream = mp.Process(target=stream, args=(queue_s2d, queue_d2s))
    process_stream.start()


    process_detect = mp.Process(target=detect, args=(queue_s2d, queue_d2s))
    process_detect.start()

    # main()
