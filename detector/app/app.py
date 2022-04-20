import multiprocessing as mp
import queue, time, os, json, time, datetime
import cv2, io, traceback, os
from flask import Flask, render_template, Response
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

VIDEO_SCREEN_SIZE = (640, 480)
# VIDEO_SCREEN_SIZE = (1920, 1080)
# VIDEO_SCREEN_SIZE = (1280, 720)

from . import ellipse, circle, mqtt

WIDTH_CIRCLE = 0.60 # meter
DIAGONAL_FOV_ANGLE_X = 78 # degree
DIAGONAL_FOV_ANGLE_y = DIAGONAL_FOV_ANGLE_X

fitting_function = ellipse

INPUT_CAPS = "udpsrc port=6000 ! " \
       "application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! " \
       "rtph264depay ! " \
       "avdec_h264 ! " \
       "videoconvert ! " \
       "video/x-raw,format=BGR ! " \
       "appsink drop=1"

# INPUT_CAPS = "v4l2src device=/dev/video0 ! video/x-raw,width=1280,height=720 ! videoscale ! videoconvert ! appsink"

mqtt_configuration = mqtt.Configuration(
    broker_ip_address="40.114.234.93",
    broker_port=1883,
    broker_publish_topic="/video_aggregates",
    broker_subscribe_topics=["/dashboard_information/live"]
)
mqtt_client = mqtt.Client(config=mqtt_configuration)
mqtt_client.connect()


def mqtt_stream(queue_m2s):

    @mqtt.ParseIncomingMessage()
    def on_message(client, user_data, message, parsed_message):
        try:
            queue_m2s.put_nowait(json.dumps(parsed_message))
        except Exception:
            pass
        
    mqtt_client.bind_on_message(on_message_function_object=on_message)
    mqtt_client.loop_forever()


class TimeIt():
    def __init__(self):
        self._process_time_1 = time.time()

    def set_time(self):
        self._process_time_0 = self._process_time_1
        self._process_time_1 = time.time()

    @property
    def process_time(self):
        try:
            return self._process_time_1 - self._process_time_0
        except Exception:
            return -1

def detect(queue_s2d, queue_d2s):
    params = cv2.SimpleBlobDetector_Params()

    # params.minThreshold = 10
    # params.maxThreshold = 200

    params.filterByArea = True
    params.minArea = 100

    # params.filterByCircularity = True
    # params.minCircularity = 0.5

    # params.filterByInertia = True
    # params.minInertiaRatio = 0.2


    detector = cv2.SimpleBlobDetector_create(params)
    
    while True:
        frame = queue_s2d.get()
        detected_height = "X"

        processing_time_0 = time.time()

        frame_mask = np.zeros_like(frame)
        processing_height_bar = ""
        did_detect = False

        keypoints = detector.detect(frame)
        number_of_blobs = len(keypoints)
        location_blobs = np.array([[keypoint.pt[0], keypoint.pt[1]] for keypoint in keypoints])

        if number_of_blobs > 20:
            center_x, center_y, ellipse_width, ellipse_height, phi = fitting_function.fit(location_blobs)
    
            half_width = max(VIDEO_SCREEN_SIZE[0] / (2 * ellipse_width), VIDEO_SCREEN_SIZE[1] / (2 * ellipse_height)) * WIDTH_CIRCLE / 2
            height = half_width / np.tan(np.pi / 180 * DIAGONAL_FOV_ANGLE_X / 2 )

            processing_time_1 = time.time()

            for i in range(number_of_blobs):
                cv2.circle(frame_mask, (int(location_blobs[i,0]), int(location_blobs[i,1])), 1, (0, 0, 255), 2)

            cv2.ellipse(
                frame_mask,
                (int(center_x), int(center_y)),
                (int(ellipse_width), int(ellipse_height)),
                phi * 180 / np.pi,
                0,
                360,
                (0, 255, 0),
                2,
            )

            cv2.drawKeypoints(
            frame_mask,
            keypoints,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
            )  
            detected_height = int(100 * height)
            processing_freq = int(1/(processing_time_1 - processing_time_0))
            processing_height_bar = processing_freq * "|"
            did_detect = True

            mqtt_message = dict(
                time_stamp = "",
                height = detected_height,
                height_method = "dots",
                number_of_detected_dots = number_of_blobs,
                distance_to_center = -1
            )

            mqtt_client.publish(
                mqtt_configuration.broker_publish_topic,
                json.dumps(mqtt_message)
            )


        cv2.putText(frame_mask, f"Height: {detected_height} cm", (10,30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0),thickness=1)
        cv2.putText(frame_mask, f"dPS: {processing_height_bar}", (10,120), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0),thickness=1)
        
        queue_d2s.put((frame_mask, did_detect))


def stream(queue_s2d, queue_d2s, queue_m2s):
    app = Flask(__name__, static_folder=os.path.abspath('./static'))

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

    @app.route('/mqtt_feed')
    def mqtt_feed():
        message = ""

        def generate():
            global message
            try: 
                message = queue_m2s.get_nowait()
            except Exception:
                pass
            return message

        return Response(generate(), mimetype='text') 

    @app.route("/video_feed")
    def video_feed():
        """Video streaming route. Put this in the src attribute of an img tag."""
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def gen():
        """Video streaming generator function."""

        frame_mask = np.zeros((VIDEO_SCREEN_SIZE[1], VIDEO_SCREEN_SIZE[0], 3))

        time_it_stream = TimeIt()
        time_it_detection = TimeIt()

        while True:
            time_it_stream.set_time()
            _, frame = vc.read()
            did_detect = False

            frame = cv2.resize(frame, VIDEO_SCREEN_SIZE)

            if queue_s2d.empty():
                queue_s2d.put(frame)

            if not queue_d2s.empty():
                frame_mask, did_detect = queue_d2s.get()
                time_it_detection.set_time()


            detection_fps_bar = int(1/time_it_detection.process_time) * did_detect * "|"
            stream_fps_bar = int(1/time_it_stream.process_time) * "|"

            cv2.putText(frame_mask, f"DPS: {detection_fps_bar}", (10,90), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0),thickness=1)
            cv2.putText(frame_mask, f"FPS: {stream_fps_bar}", (10,60), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0, 255, 0),thickness=1)

            frame_bool_mask = frame_mask > 0
            frame[frame_bool_mask] = frame_mask[frame_bool_mask]

            _, image_buffer = cv2.imencode(".jpg", frame)
            io_buf = io.BytesIO(image_buffer)

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + io_buf.read() + b"\r\n"
            )

    with VideoCapture(INPUT_CAPS, cv2.CAP_GSTREAMER) as vc:
        app.run(host="0.0.0.0", threaded=True)


if __name__ == "__main__":
    queue_s2d = mp.Queue()
    queue_d2s = mp.Queue()

    queue_m2s = mp.Queue(1)

    process_stream = mp.Process(target=stream, args=(queue_s2d, queue_d2s, queue_m2s))
    process_stream.start()

    mqtt_stream = mp.Process(target=mqtt_stream, args=(queue_m2s,))
    mqtt_stream.start()

    detection_threads = []
    NUMBER_OF_DETECTION_THREADS = int(os.environ.get("NUMBER_OF_DETECTION_THREADS", 1))

    for i in range(NUMBER_OF_DETECTION_THREADS):
        process_detect = mp.Process(target=detect, args=(queue_s2d, queue_d2s))
        detection_threads.append(process_detect)
        detection_threads[i].start()
