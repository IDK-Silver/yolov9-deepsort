import pathlib
import random
from typing import List, Union

import numpy as np
import torch
import cv2
import typing

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

from .yolov9.models.utils.augmentations import letterbox
from .yolov9.models.utils.torch_utils import select_device
from .yolov9.models.common import DetectMultiBackend
from .yolov9.models.utils.general import check_img_size, non_max_suppression, scale_boxes

# for GTX 1650 ti
torch.backends.cudnn.enabled = False


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        font_size = t_size[1]

        font = ImageFont.truetype(
            str(pathlib.Path(__file__).parent / "fonts" / "NotoSansTC-Regular.otf"),
            font_size, encoding="utf-8"
        )

        # in Pillow 10, using this method to get font size
        font_box = font.getbbox(label)
        t_size = (font_box[2], font_box[3])
        c2 = c1[0] + t_size[0], c1[1] - t_size[1]
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        draw.text((c1[0], c2[1]), label, (255, 255, 255), font=font)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class YOLOv9_Detector:
    def __init__(
            self, conf: float = 0.25, conf_threshold: float = 0.25,
            iou_threshold: float = 0.45, classes: list = None,
            names: List[str] = None
    ):
        self.device = select_device(
            "0" if torch.cuda.is_available()
            else 'cpu'
        )
        self.conf = conf
        self.model: Union[torch.Module, DetectMultiBackend, None] = None
        self.image_size = (640, 640)
        self.model_classify = None
        self.names: Union[list, str] = names
        self.colors: list = []
        self.half_precision = False
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.stride = None

    def load_model(self, model_path):
        # whether device can use half precision or not
        self.half_precision = self.device.type != 'cpu'

        # load model
        self.model = DetectMultiBackend(
            weights=model_path, device=self.device, dnn=False, fp16=self.half_precision
        )

        self.stride, self.names, pt = self.model.stride, self.model.names, self.model.pt

        self.image_size = check_img_size(imgsz=self.image_size, s=self.stride)

        # pred one to check model
        self.model.warmup(imgsz=(1 if pt or self.model.triton else 1, 3, *self.image_size))

        self.colors = [
            [random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))
        ]

        print(f'Loaded names by model: {self.names}')

    def detect(self, original_image, plot=False):

        image, original_image = self.load_image(original_image)

        image = torch.from_numpy(image).to(self.device)
        image = image.half() if self.half_precision else image
        image /= 255.0

        if len(image.shape) == 3:
            image = image[None]

        pred = self.model(image, augment=False)
        detection = non_max_suppression(pred, self.conf, self.iou_threshold, classes=None)

        detection = detection[0]

        if len(detection):
            # rescale boxes from img_size to im0 size
            # https://blog.csdn.net/m0_46483236/article/details/123605527
            detection[:, :4] = scale_boxes(
                image.shape[2:], detection[:, :4], original_image.shape
            ).round()

            for *det_points, det_conf, det_cls in reversed(detection):
                if plot:
                    label = f'{self.names[int(det_cls)]} {det_conf:.2f}'

                    original_image = plot_one_box(
                        det_points, original_image, label=label,
                        color=self.colors[int(det_cls)], line_thickness=2
                    )

            if plot:
                return original_image
            else:
                return detection.detach().cpu().numpy()

        return original_image if plot else None

    def load_image(self, original_image: typing.Union[str, np.array]):

        # if image is file path, loading image
        if isinstance(original_image, str):
            original_image = cv2.imread(original_image)
        assert original_image is not None, 'Image Not Found '

        # letterbox https://medium.com/mlearning-ai/letterbox-in-object-detection-77ee14e5ac46
        image: np.ndarray = letterbox(
            original_image,
            self.image_size,
            stride=self.stride
        )[0]

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # to 3 x width x height
        # https://zhuanlan.zhihu.com/p/61203757
        image = image.transpose(2, 0, 1)

        # make matrix is continuous in memory
        # https://zhuanlan.zhihu.com/p/59767914
        image = np.ascontiguousarray(image)

        return image, original_image
