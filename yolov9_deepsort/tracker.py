import pathlib
from typing import Union
import torch
import cv2
import numpy as np

from .detector import YOLOv9_Detector
from .tools.generate_detections import create_box_encoder
from .deep_sort.tracker import Tracker
from .deep_sort.track import Track
from .deep_sort.detection import Detection
from .deep_sort.nn_matching import NearestNeighborDistanceMetric
from .detector import plot_one_box

# for GTX 1650 ti
torch.backends.cudnn.enabled = False


# https://zhuanlan.zhihu.com/p/354937153
class YOLOv9_DeepSORT:

    def __init__(
            self, detector: YOLOv9_Detector, reid_model_path: str = None,
            max_cosine_distance: float = 0.4, nn_budget: float = None
    ):
        if reid_model_path is None:
            reid_model_path = str(pathlib.Path(__file__).parent / 'deep_sort' / 'model_weights' / 'mars-small128.pb')

        self.detector = detector
        self.names: list = detector.names
        self.encoder = create_box_encoder(reid_model_path, batch_size=1)
        # calculate cosine distance metric
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def track_video(self, video_source: Union[str, int]):
        # load video

        # OpenCV https://tinyurl.com/26vxufnv
        if isinstance(video_source, str):
            video = cv2.VideoCapture(video_source)
        else:
            video = cv2.VideoCapture(int(video_source))

        if not video.isOpened():
            print('video init error')
            exit(-1)

        frame_num: int = 0
        while True:

            reval, frame = video.read()

            if not reval:
                print('video has ended or error', 'skip this frame')
                break

            frame_num += 1

            # [x,y,w,h, confidence, class]
            detect_result = self.detector.detect(frame.copy(), plot=False)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # points_datas: List[List[int]] = []
            # conf_datas: List[int] = []
            # cls_datas: List[int] = []
            # num_objects: int = 0

            if detect_result is not None:
                # get points datas
                points_datas = detect_result[:, :4]

                # change [x1, y1, x2, y2] to [x1, y1, width, height]
                points_datas[:, 2] = points_datas[:, 2] - points_datas[:, 0]
                points_datas[:, 3] = points_datas[:, 3] - points_datas[:, 1]

                conf_datas = detect_result[:, 4]
                cls_datas = detect_result[:, -1]
                num_objects = points_datas.shape[0]
            else:
                points_datas = []
                conf_datas = []
                cls_datas = []
                num_objects = 0

            labels = []

            for index in range(num_objects):
                labels.append(
                    self.names[
                        int(cls_datas[index])
                    ]
                )

            features = self.encoder(frame, points_datas)

            deepsort_detections = []
            for points, conf, cls, feature in zip(points_datas, conf_datas, cls_datas, features):
                deepsort_detections.append(
                    Detection(points, conf, feature, cls)
                )

            colors = self.detector.colors

            # DeepSORT predict
            self.tracker.predict()
            self.tracker.update(deepsort_detections)

            track: Track
            for track in self.tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_points = track.to_tlbr()
                track_cls = track.get_class()

                color = colors[int(track.track_id) % len(colors)]

                frame = plot_one_box(
                    track_points, frame, label=(self.names[int(track_cls)] + ', id ' + str(int(track.track_id))),
                    color=color, line_thickness=2
                )

            result = cv2.cvtColor(
                np.asarray(frame), cv2.COLOR_RGB2BGR
            )

            cv2.imshow('DeepSORT Result', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
