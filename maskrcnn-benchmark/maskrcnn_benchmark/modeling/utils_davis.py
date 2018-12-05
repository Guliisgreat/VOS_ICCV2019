from davis_components.inference import pad_box
import torch
import numpy as np


def pad_boxes_on_detections(detections, pad_size):
    for detection in detections:
        boxes = detection.bbox.tolist()
        shape = (detection.get_img_height(),
                 detection.get_img_width())
        detection.bbox = torch.tensor(np.array(pad_box(boxes, pad_size, shape, type="list")),
                                      dtype=torch.float32, device=torch.device("cuda")).reshape((-1, 4))
    return detections

