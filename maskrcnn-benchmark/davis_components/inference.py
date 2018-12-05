import cv2
import torch
import numpy as np
from torchvision import transforms as T


def select_top_predictions(predictions, confidence_threshold=0.7):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score

    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.

    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores >  confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]


def overlay_boxes(image, predictions):
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


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors


def vote_pixel_of_mask_for_annotation(masks, prediction, threshold=0.5):
    """
    Each prediction(box) possiblely overlaps with others,  but there is only one label
    to represent each instance's pixels in the final annotation.
    Thus, we need to decide which instance each pixel belongs to. For each pixel,
    we define a pixel score for each instance, and choose the instance with
    the maximum pixel score as its pixel's label

    For each pixel:
    Pixel Score = foreground probability in the mask * its mask's class probability
    (In DAVIS, there is only 2 categories)

    Arguments:
        prediction (BoxList): the result of the computation by the model.
            It should contain the field `scores`.'label'
        masks (torch.tensor):  a set of masks (prediction.get_field("mask"))have been projected
            in an image on the locations specified by the bounding boxes
        threshold (float): the threshold to decide whether the pixel is the
            foreground or background

    Returns:
        annotation (torch.tenor): the final annotation  included with all instances
    """
    num_instance = len(masks)

    scores = prediction.get_field("scores").numpy()
    masks = masks.numpy()

    threshold = adaptive_threshold(masks, threshold)


    instance_scores = np.array([score * mask for score, mask in zip(scores, masks)])
    instance_scores = ((masks > threshold) + 0) * instance_scores
    background_scores = np.expand_dims(np.zeros(np.array(instance_scores).shape[1:]), axis=0)
    pixel_scores = np.concatenate((background_scores, instance_scores), axis=0)
    annotation = np.array(pixel_scores).argmax(axis=0)

    if prediction.has_field("instance_id"):
        instance_ids = prediction.get_field("instance_id").numpy()
        annotation = assign_template_instance_id(annotation, instance_ids)

    if len(annotation.shape) == 3:
        annotation = np.expand_dims(np.array(annotation, dtype=np.uint8), axis=0)
    annotation = torch.from_numpy(annotation)
    return annotation


def assign_template_instance_id(annotation, instance_ids):
    obj_list = list(np.unique(annotation))
    obj_list.pop(0)
    # if len(obj_list) != len(instance_ids):
    #     a =1
    # assert len(obj_list) == len(instance_ids), \
    #     "the number of objects in the prediction " \
    #     "is not matched with that in the template"
    for obj in obj_list:
        instance_id = instance_ids[obj-1]
        np.place(annotation, annotation == obj, instance_id)
    return annotation


def adaptive_threshold(masks, threshold):
    """
        Make sure each instance will not removed because of its low score,
         and will show in the final annotation

    """
    for mask in masks:
        if mask.max() < threshold:
            threshold = mask.max()
    return threshold


def pad_box(boxes, pad_size, shape, type="list"):
    """
        Increase the area of bounding boxes by padding. Support a batch of boxes

        Arguments:
            boxes(list): mode = "xyxy"
            pad_size (int):
            shape (int): the shape of original shape, which determines
                        the max and min of bounding boxes
            type(str): list or tensor
                     tensor means each bounding box should be torch.tensor
        Return:
            boxes(list): mode = "xyxy"
    """
    assert isinstance(boxes, list), "the type of boxes is incorrect"
    num_box = len(boxes)
    padded_box = []
    for box in boxes:
        x_max = shape[1]
        y_max = shape[0]
        if type == "tensor":
            new_box = torch.tensor([max(box[0] - pad_size, 0),
                        max(box[1] - pad_size, 0),
                        min(box[2] + pad_size, x_max),
                        min(box[3] + pad_size, y_max)])
        else:
            new_box = [max(box[0] - pad_size, 0),
                        max(box[1] - pad_size, 0),
                        min(box[2] + pad_size, x_max),
                        min(box[3] + pad_size, y_max)]
        padded_box.append(new_box)
    assert num_box == len(padded_box), "the num of boxes after padding != before"
    return padded_box

