import multiprocessing as mp
import cv2, io, traceback, logging, os
from flask import Flask, render_template, Response
import numpy as np

VIDEO_SCREEN_SIZE = (640, 480)

def main():
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

            if frame is None:
                logging.warning(
                    "Frame is of type NoneType, -> error with /dev/usb0 -> reset Raspberry..."
                )

            _, image_buffer = cv2.imencode(".jpg", frame)
            io_buf = io.BytesIO(image_buffer)

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + io_buf.read() + b"\r\n"
            )

    with VideoCapture(0) as vc:
        app.run(host="0.0.0.0", threaded=True)

if __name__ == "__main__":
    main()
