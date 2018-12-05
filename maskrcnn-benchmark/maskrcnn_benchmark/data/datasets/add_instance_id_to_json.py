from pycocotools.coco import COCO
import collections
from PIL import Image
import os
import numpy as np
import json
from tqdm import tqdm
from utils_davis.annotation import boxes_for_on_annotation
from utils_davis.evaluation import calc_iou_individual


def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return listOfKeys


def xywh_to_xyxy(box):
    box[2] = box[0] + box[2]
    box[3] = box[1] + box[3]
    box = [int(_) for _ in box]
    return box

def pad_box(box, pad_size, shape):
    x_max = shape[1]
    y_max = shape[0]
    box[0] = max(box[0] - pad_size, 0)
    box[1] = max(box[1] - pad_size, 0)
    box[2] = min(box[2] + pad_size, x_max)
    box[3] = min(box[3] + pad_size, y_max)
    return box


def check_instance_id_unique(imgToAnno):
    if len(imgToAnno) == 0:
        raise Exception

    instance_id_list = [instance['instance_id'] for instance in imgToAnno]
    if len(imgToAnno) == len(list(np.unique(instance_id_list))):
        return True
    else:
        return False


def find_instance_id(box, template_boxes):
    iou_list = []
    for template_box in template_boxes:
        iou_list.append(calc_iou_individual(box, template_box['box']))
    idx = np.argmax(np.array(iou_list))
    return template_boxes[idx]["instance_id"]






def main():
    root = '/home/guli/Desktop/VOS_ICCV2019/maskrcnn-benchmark/datasets/DAVIS'
    annFile = "/home/guli/Desktop/VOS_ICCV2019/maskrcnn-benchmark/datasets/DAVIS/Annotations/instancesonly_480p_val.json"
    coco = COCO(annFile)
    # with open(annFile) as json_data:
    #     coco = json.load(json_data)
    imgs = coco.imgs
    imgToAnns = coco.imgToAnns
    anns = coco.anns


    new_file = collections.defaultdict(list)
    new_file['categories'].append(coco.dataset['categories'][0])
    new_file['images'] = coco.dataset['images']

    for idx, key in tqdm(enumerate(imgs.keys())):
        img = imgs[key]
        annotation = np.array(Image.open(os.path.join(root, img['seg_file_name'])))

        # if idx > 20:
        #     break
        boxes_list = boxes_for_on_annotation(annotation, )

        for instance in imgToAnns[key]:
            memory = []
            if len(imgToAnns[key]) == 3:
                a = 1
            box = xywh_to_xyxy(instance['bbox'])
            box = pad_box(box, pad_size=0, shape=annotation.shape)

            instance_id = find_instance_id(box, boxes_list)

            # mask = annotation[box[1]:box[3], box[0]:box[2]]
            # counter_dict = collections.Counter(list(mask.flatten()))
            # result = sorted(counter_dict.values())
            # if len(result) == 1:
            #     a = 1
            # instance_id = getKeysByValue(counter_dict, result[-1])
            # if instance_id[0] == 0:
            #     instance_id = getKeysByValue(counter_dict, result[-2])
            #
            # if instance_id[0] not in memory:
            #     memory.append(instance_id)
            # else:
            #     raise Exception

            anns[instance['id']]['instance_id'] = np.int(instance_id)

            new_file['annotations'].append(anns[instance['id']])

        if not check_instance_id_unique(imgToAnns[key]):
            raise Exception



    filename = "/home/guli/Desktop/VOS_ICCV2019/maskrcnn-benchmark/" \
              "datasets/DAVIS/Annotations/instancesonly_480p_val_instance_id.json"
    with open(filename, 'w') as outfile:
        json.dump(dict(new_file), outfile)
        print("Write file finished.")

    # with open(filename, 'r') as json_data:
    #     p = json.load(json_data)
    #     a = 1

def unit_test():
    root = '/home/guli/Desktop/VOS_ICCV2019/maskrcnn-benchmark/datasets/DAVIS'
    annFile = "/home/guli/Desktop/VOS_ICCV2019/maskrcnn-benchmark/" \
              "datasets/DAVIS/Annotations/instancesonly_480p_val_instance_id.json"

    coco = COCO(annFile)
    a = 1

# images
# annotations
# categories

if __name__ == '__main__':

    main()
    # unit_test()