import os
from PIL import Image
import numpy as np

import torch
import torchvision
import maskrcnn_benchmark.data.datasets.coco as coco
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class DAVISDatasetAddInstanceID(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(DAVISDatasetAddInstanceID, self).__init__(root, ann_file)

        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms_davis = transforms

    def __getitem__(self, idx):
        img, anno = super(DAVISDatasetAddInstanceID, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes

        target = BoxList(boxes, img.size, mode="xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        num_instance = len(anno)
        instance_ids = [obj["instance_id"] for obj in anno]
        # if num_instance != len(list(np.unique(instance_ids))):
        #     a = 1
        # assert num_instance == len(list(np.unique(instance_ids))), "number of instance_id error "
        instance_ids = torch.as_tensor(instance_ids)
        target.add_field("instance_id", instance_ids)


        target = target.clip_to_image(remove_empty=True)

        if self.transforms_davis is not None:
            img, target = self.transforms_davis(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

    def get_img(self, index):
        img_id = self.id_to_img_map[index]
        img_data = np.array(Image.open(\
            os.path.join(self.root, self.coco.imgs[img_id]["file_name"])))
        return img_data

    def get_annotation(self, index):
        img_data = self.get_img_info(index)
        seg_file_name = img_data["seg_file_name"]
        annotation_data = np.array(Image.open(\
            os.path.join(self.root, seg_file_name)))
        return annotation_data

    def get_annotation_box(self, index):
        img_id = self.id_to_img_map[index]
        anns_id_list = self.coco.imgToAnns[img_id]
        boxes_list_xywh = [_['bbox']for _ in anns_id_list]
        boxes_list_xyxy = [(box[0], box[1], box[0]+box[2], box[1]+box[3]) for box in boxes_list_xywh]
        return boxes_list_xyxy

    def get_img_width(self, index):
        img_id = self.id_to_img_map[index]
        return self.coco.imgs[img_id]['width']

    def get_img_height(self, index):
        img_id = self.id_to_img_map[index]
        return self.coco.imgs[img_id]['height']

    def get_annotation_video_id(self, index):
        img_data = self.get_img_info(index)
        seg_file_name = img_data["seg_file_name"]
        video_id = seg_file_name.split('/')[-2]
        return video_id

    def get_annotation_img_id(self, index):
        img_data = self.get_img_info(index)
        seg_file_name = img_data["seg_file_name"]
        image_id = seg_file_name.split('/')[-1].split('.')[-2]
        return image_id

    def get_annotation_palette(self, index):
        img_data = self.get_img_info(index)
        seg_file_name = img_data["seg_file_name"]
        palette = Image.open(os.path.join(self.root, seg_file_name)).getpalette()
        return palette


class DAVISDataset(coco.COCODataset):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(DAVISDataset, self).__init__(ann_file, root, remove_images_without_annotations)
        self.transforms_davis = transforms

    def __getitem__(self, idx):
        img, target, idx = super(DAVISDataset, self).__getitem__(idx)
        if self.transforms_davis is not None:
            img, target = self.transforms_davis(img, target)

        return img, target, idx

    def get_img(self, index):
        img_id = self.id_to_img_map[index]
        img_data = np.array(Image.open(\
            os.path.join(self.root, self.coco.imgs[img_id]["file_name"])))
        return img_data

    def get_annotation(self, index):
        img_data = self.get_img_info(index)
        seg_file_name = img_data["seg_file_name"]
        annotation_data = np.array(Image.open(\
            os.path.join(self.root, seg_file_name)))
        return annotation_data

    def get_annotation_box(self, index):
        img_id = self.id_to_img_map[index]
        anns_id_list = self.coco.imgToAnns[img_id]
        boxes_list_xywh = [_['bbox']for _ in anns_id_list]
        boxes_list_xyxy = [(box[0], box[1], box[0]+box[2], box[1]+box[3]) for box in boxes_list_xywh]
        return boxes_list_xyxy

    def get_img_width(self, index):
        img_id = self.id_to_img_map[index]
        return self.coco.imgs[img_id]['width']

    def get_img_height(self, index):
        img_id = self.id_to_img_map[index]
        return self.coco.imgs[img_id]['height']

    def get_annotation_video_id(self, index):
        img_data = self.get_img_info(index)
        seg_file_name = img_data["seg_file_name"]
        video_id = seg_file_name.split('/')[-2]
        return video_id

    def get_annotation_img_id(self, index):
        img_data = self.get_img_info(index)
        seg_file_name = img_data["seg_file_name"]
        image_id = seg_file_name.split('/')[-1].split('.')[-2]
        return image_id

    def get_annotation_palette(self, index):
        img_data = self.get_img_info(index)
        seg_file_name = img_data["seg_file_name"]
        palette = Image.open(os.path.join(self.root, seg_file_name)).getpalette()
        return palette
