import numpy as np


def boxes_for_on_annotation(annotation):
    '''
    1. Find the number of objects in this annotation
    2. Acquire the bbox for each objects
    :param annotation:
    :return: bbox_list: including all objects' bbox in this one annotation
                        [object_idx, bbox]
    '''
    annotation[annotation == 255] = 0  # 255 for object in the middle of video frame
    objs = np.unique(annotation)

    bbox_list = []
    for obj in objs:
        result = {}
        if obj == 0:
            continue
        bbox = coordinate_mask2bbox(annotation, obj)
        result["box"] = bbox
        result["instance_id"] = obj
        bbox_list.append(result)

    return bbox_list


def coordinate_mask2bbox(annotation, idx_object):
    """
    :param annotation: the mask included all detected objects' masks, where using the index of  each object.
                 e.g. If there are three objects in this mask, the background is 0 and other object's mask are indexed with 1, 2 and 3
    :param idx_object:

    :return: coordinate_bbox: [min_x, min_y, max_x, max_y]
    """
    object_mask = (annotation == idx_object)
    object_coords = np.argwhere(object_mask)

    y0, x0 = object_coords.min(axis=0)
    y1, x1 = object_coords.max(axis=0) + 1
    coordinate_bbox = [x0, y0, x1, y1]

    return coordinate_bbox

