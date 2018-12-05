import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.bounding_box import BoxList


def select_top_boxes_from_prediction(predictions, confidence_threshold=0.7):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.
        confidence_threshold: threshold for object score

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]


def overlay_boxes(image, predictions):
    if isinstance(predictions, BoxList):
        return overlay_boxes_from_boxlist(image, predictions)
    elif isinstance(predictions, list):
        return overlay_boxes_from_list(image, predictions)
    else:
        raise Exception


def overlay_boxes_from_boxlist(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image


def overlay_boxes_from_list(image, boxes):
    """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            boxes (list): all boxes on this image (xyxy).
        """
    for box in boxes:
        box = np.array(box).astype(int)
        top_left, bottom_right = box[:2], box[2:]
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), 1)
    return image


def overlay_one_box_on_annotation(annotation, box):
    """
        Add one predicted box on top of the annotation.
        Usually used for debugging to check the accuracy of the box

        Arguments:
              annotation (np.ndarray)
              box(list): (xyxy)

    """
    assert len(box) == 4, "the format of the box is incorrect!"
    assert len(annotation.shape) == 2, "the format of the annotation is incorrect!"
    box = np.array(box).astype(int)
    top_left, bottom_right = box[:2], box[2:]
    annotation = cv2.rectangle(
        annotation, tuple(top_left), tuple(bottom_right), (256,256,256), 1)
    return annotation




def overlay_mask( image, predictions):
    """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    masks = predictions.get_field("mask").numpy()
    labels = predictions.get_field("labels")

    colors = compute_colors_for_labels(labels).tolist()

    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None]
        _, contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        image = cv2.drawContours(image, contours, -1, color, 3)

    composite = image

    return composite


def create_mask_montage(image, predictions):
    """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
        It should contain the field `mask`.
    """
    masks = predictions.get_field("mask")
    masks_per_dim = 2
    masks = torch.nn.functional.interpolate(
        masks.float(), scale_factor=1 / masks_per_dim
    ).byte()
    height, width = masks.shape[-2:]
    max_masks = masks_per_dim ** 2
    masks = masks[:max_masks]
    # handle case where we have less detections than max_masks
    if len(masks) < max_masks:
        masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
        masks_padded[: len(masks)] = masks
        masks = masks_padded
    masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
    result = torch.zeros(
        (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
    )
    for y in range(masks_per_dim):
        start_y = y * height
        end_y = (y + 1) * height
        for x in range(masks_per_dim):
            start_x = x * width
            end_x = (x + 1) * width
            result[start_y:end_y, start_x:end_x] = masks[y, x]
    return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors


def visualize_batch_mask_for_debug(masks, dirname):
    """
        Visualize the mask or annotation and save results into directionary
        Support several format (np.ndarray, (torch.tenor))

        Arguments:
        masks (np.ndarray, torch.Tensor): a batch of masks or a batch of annotations
        dirname: output's folder
    """
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().detach().numpy()

    if len(masks.shape) == 2:
        batch_size = 1
    elif len(masks.shape) == 3:
        batch_size = masks.shape[0]
    elif len(masks.shape) == 4 and masks.shape[1] == 1:
        batch_size = masks.shape[0]
    else:
        raise ValueError('the shape of masks is incorrect')

    if batch_size > 1:
        for idx, mask in enumerate(masks):
            # Remove 1 in (batch_size, 1, height, width)
            if len(mask.shape) == 3:
                mask = mask.squeeze(axis=0)
            filename = os.path.join(dirname, str(idx) + '.png')
            plt.imshow(mask)
            plt.savefig(filename)
    else:
        filename = os.path.join(dirname, '0.png')
        cv2.imwrite(filename, masks)


def save_img_visualization(result, dirname, video_id, img_id):
    file = os.path.join(dirname, video_id)
    if not os.path.exists(file):
        os.mkdir(file)
    filename = os.path.join(file, img_id) + ".png"
    cv2.imwrite(filename, result*255)
    # plt.imshow(result)
    # plt.savefig(filename)

