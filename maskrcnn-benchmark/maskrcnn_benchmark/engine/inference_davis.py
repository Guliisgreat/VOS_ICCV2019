# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import tempfile
import time
import os
from collections import OrderedDict

import torch

from tqdm import tqdm
from PIL import Image
import numpy as np

from ..structures.bounding_box import BoxList
from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize


from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from davis_components.inference import vote_pixel_of_mask_for_annotation

from instanceMatching.matching import InstanceMatcher
from utils_davis.visualization import overlay_boxes, select_top_boxes_from_prediction, save_img_visualization
from utils_davis import tools
from utils_davis.evaluation import davis_toolbox_evaluation


def compute_on_dataset(model, data_loader, device, debug):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in tqdm(enumerate(data_loader)):
        if i == 181:
            a =1

        if debug:
            if i > 10:
                break

        images, targets, image_ids = batch
        images = images.to(device)
        with torch.no_grad():
            output = model(images)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def compute_on_dataset_with_gt(model, data_loader, device, debug):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for i, batch in tqdm(enumerate(data_loader)):
        if i == 181:
            a =1

        if debug:
            if i > 10:
                break

        images, targets, image_ids = batch
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            output = model(images, targets)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def matching_instance_with_gt_template(data_loader, predictions, debug):
    matcher = InstanceMatcher(iou_type='bbox')
    matched_prediction = []

    not_detect_counter = 0

    for i, batch in tqdm(enumerate(data_loader)):


        # debug part
        dataset = data_loader.dataset
        video_id = dataset.get_annotation_video_id(i)
        img_id = dataset.get_annotation_img_id(i)
        if video_id == "bmx-trees" and img_id == "00000":
            a = 1

        if debug:
            if i > 10:
                break
        batch_size = len(batch[1])
        images, targets, image_ids = batch
        templates_batch = targets
        predictions_batch = predictions[i*batch_size:i*batch_size+batch_size]

        assert len(predictions_batch) == len(templates_batch), \
            "the batch_size of templates and predictions are not matched"
        for prediction, template in zip(predictions_batch, templates_batch):
            # debug
            image_width = prediction.get_img_width()
            image_height = prediction.get_img_height()

            template = template.resize((image_width, image_height))
            template = template.convert('xyxy')

            #
            if len(prediction) < len(template):
                # matched_prediction.append(template)
                not_detect_counter += 1
                print("video_id: {}, img_id {}, not_detect_counter = {}".format(video_id, img_id, not_detect_counter))



            matched_prediction.append(matcher(prediction, template))

    return matched_prediction


def prepare_for_davis_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    davis_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.get_img_width(image_id)
        image_height = dataset.get_img_height(image_id)
        prediction = prediction.resize((image_width, image_height))
        # prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        davis_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return davis_results


def prepare_for_davis_segmentation(predictions, dataset):
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # TODO replace with get_img_info?
        image_width = dataset.get_img_width(image_id)
        image_height = dataset.get_img_height(image_id)
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        masks = masker(masks, prediction)
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # rles = prediction.get_field('mask')

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results


def check_prediction_without_instance(predictions):
    checked_prediction = []
    for id, prediction in tqdm(enumerate(predictions)):
        if len(prediction) == 0:
            prediction = last_prediction
        checked_prediction.append(prediction)
        last_prediction = prediction
    return checked_prediction


def resize_predictions(predictions, dataset):
    resized_predictions = []
    for id, prediction in tqdm(enumerate(predictions)):
        image_width = dataset.get_img_width(id)
        image_height = dataset.get_img_height(id)

        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xyxy')
        resized_predictions.append(prediction)
    return resized_predictions

def generate_davis_final_annotation(predictions, dataset, output_file, annotation_type):
    masker = Masker(padding=1, keep_score=True)
    # assert isinstance(dataset, COCODataset)
    davis_results = []
    for id, prediction in tqdm(enumerate(predictions)):
        if len(prediction) == 0:
            continue

        video_id = dataset.get_annotation_video_id(id)
        img_id = dataset.get_annotation_img_id(id)

        if video_id == "scooter-black" and img_id == "00012":
            a = 1

        palette = dataset.get_annotation_palette(id)
        image_width = dataset.get_img_width(id)
        image_height = dataset.get_img_height(id)

        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert('xyxy')
        masks = prediction.get_field("mask")
        # visualize_batch_mask_for_debug(masks, output_file)
        masks = masker(masks, prediction)
        # save_final_annotation(masks, video_id, img_id, palette, output_file)

        final_predict_annotation = vote_pixel_of_mask_for_annotation(masks, prediction, threshold=0.5)
        # Vote
        # annotation = ((annotation > 0.5) + 0).astype(int)
        save_final_annotation(final_predict_annotation, video_id, img_id, palette, output_file, annotation_type)


def save_final_annotation(annotation, video_id, img_id, palette, dirname, annotation_type="no_match"):
    dirname = os.path.join(dirname, annotation_type)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    dir_video = os.path.join(dirname, video_id)
    if not os.path.exists(dir_video):
        os.mkdir(dir_video)

    annotation =  annotation.numpy()
    if not save_batch_annotations(annotation):
        file_name = os.path.join(dir_video, img_id + '.png')
        annotation = annotation.squeeze(axis=0)
        save_one_annotation(annotation, file_name, palette)
    else:
        for idx, one_annotation in enumerate(annotation):
            file_name = os.path.join(dir_video, img_id + "_" + str(idx) + '_.png')
            save_one_annotation(one_annotation, file_name, palette)


def save_batch_annotations(annotation):
    if len(annotation.shape) == 4 and annotation.shape[0] == 1:
        return False
    elif len(annotation.shape) == 4 and annotation.shape[0] > 1:
        return True
    else:
        raise Exception


def save_one_annotation(annotation, filename, palette):
    assert len(annotation.shape) == 3, \
        "the shape of one annotation is not correct!"
    annotation = annotation.squeeze(axis=0)
    result = Image.fromarray(annotation.astype(np.uint8), 'P')
    result.putpalette(palette)
    result.save(filename)




# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        # TODO replace with get_img_info?
        image_width = dataset.coco.imgs[original_id]["width"]
        image_height = dataset.coco.imgs[original_id]["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("scores").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_predictions_on_coco(
    coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from pycocotools.cocoeval import COCOeval

    coco_dt = coco_gt.loadRes(str(json_result_file))
    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def evaluate_predictions_on_davis(
    coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from DAVIS_envaluation.davis_eval import DAVISEval, COCOeval

    coco_dt = coco_gt.loadRes(str(json_result_file))
    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = DAVISEval(coco_gt, coco_dt, iou_type)
    #
    # coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def overlay_boxes_on_predictions(predictions, dataset, dirname):
    for image_id, prediction in tqdm(enumerate(predictions)):
        if len(prediction) == 0:
            continue

        # prediction = select_top_boxes_from_prediction(prediction)

        # img = dataset.get_img(image_id)
        annotation = dataset.get_annotation(image_id)
        image_width = dataset.get_img_width(image_id)
        image_height = dataset.get_img_width(image_id)
        video_id= dataset.get_annotation_video_id(image_id)
        img_id = dataset.get_annotation_img_id(image_id)

        # prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xyxy")
        # prediction_1 = prediction.convert("xywh")
        #
        # print('-' * 30)
        # print(video_id,  img_id)
        # print('Prediction bbox:')
        # print(prediction.bbox)
        # print('gt bbox:')
        # a = dataset.get_annotation_box(image_id)
        # print(a)
        # y = []
        # for x in a:
        #     x[2] = x[0] + x[2]
        #     x[3] = x[1] + x[3]
        #     y.append(x)

        # prediction = prediction.convert("xywh")
        #
        # result = overlay_boxes(img, prediction)
        result = overlay_boxes(annotation, prediction)

        save_img_visualization(result, dirname, video_id, img_id)


def select_top_predictions(predictions, confidence_score=0.7):
    select_prediction = []
    for image_id, prediction in tqdm(enumerate(predictions)):
        if len(prediction) == 0:
            continue
        prediction = select_top_boxes_from_prediction(prediction, confidence_score)
        select_prediction.append(prediction)
    return select_prediction


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoint": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


class DAVISResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoint": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in DAVISResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, davis_eval):
        if davis_eval is None:
            return
        from DAVIS_envaluation.davis_eval import DAVISEval, COCOeval

        assert isinstance(davis_eval, DAVISEval)
        # assert isinstance(davis_eval, COCOeval)
        s = davis_eval.stats
        iou_type = davis_eval.params.iouType
        res = self.results[iou_type]
        metrics = DAVISResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)


def inference_davis(
    model,
    data_loader,
    iou_types=("bbox",),
    box_only=False,
    device="cuda",
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
    debug=True,
    generate_annotation=False,
    overlay_box=False,
    matching=False,
    skip_computation_network=False,
    select_top_predictions_flag = False,

):

    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.deprecated.get_world_size()
        if torch.distributed.deprecated.is_initialized()
        else 1
    )
    logger = logging.getLogger("DAVIS_MaskRCNN_baseline_test")
    dataset = data_loader.dataset
    if not skip_computation_network:
        logger.info("Start evaluation on {} images".format(len(dataset)))
        start_time = time.time()
        predictions = compute_on_dataset_with_gt(model, data_loader, device, debug)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        logger.info(
            "Total inference time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )

        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
        if not is_main_process():
            return

        if output_folder:
            torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
        predictions_coco = predictions

    if skip_computation_network:
        filename = os.path.join(output_folder, "predictions.pth")
        predictions = \
            torch.load(filename)
        predictions_coco = predictions

    predictions = check_prediction_without_instance(predictions)
    predictions = resize_predictions(predictions, dataset)

    if select_top_predictions_flag:
        with tools.TimerBlock("Select boxes ...", logger) as block:
            predictions = select_top_predictions(predictions, confidence_score=0.3)
            torch.save(predictions, os.path.join(output_folder, "select_predictions.pth"))

    if overlay_box:
        with tools.TimerBlock("Overlay boxes...", logger) as block:
            dirname = os.path.join(output_folder, "overlay_box_results_no_matched")
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            overlay_boxes_on_predictions(predictions, dataset, dirname)
    #
    # if generate_annotation:
    #     with tools.TimerBlock("Generate final no_matched annotation...", logger) as block:
    #         generate_davis_final_annotation(predictions, dataset, output_folder, annotation_type="no_match_0.7")

    skip_matching = False
    if matching:
        if not skip_matching:
            with tools.TimerBlock("Matching Instances...", logger) as block:
                # debug

                matched_predictions = matching_instance_with_gt_template(data_loader, predictions, debug)
                if output_folder:
                    torch.save(matched_predictions, os.path.join(output_folder, "matched_predictions.pth"))
        else:
            filename = os.path.join(output_folder, "matched_predictions.pth")
            matched_predictions = \
                torch.load(filename)

        if generate_annotation:
            with tools.TimerBlock("Generate final annotation...", logger) as block:
                matched_predictions = check_prediction_without_instance(matched_predictions)
                generate_davis_final_annotation(matched_predictions, dataset, output_folder, annotation_type="matched")



    if overlay_box:
        with tools.TimerBlock("Overlay boxes...", logger) as block:
            dirname = os.path.join(output_folder, "overlay_box_results_matched")
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            overlay_boxes_on_predictions(matched_predictions, dataset, dirname)

    with tools.TimerBlock("Using DAVIS toolbox to evaluate segmentation (J, F)...", logger) as block:
        if output_folder:
            davis_toolbox_evaluation(os.path.join(output_folder, "matched"))

    with tools.TimerBlock("Using COCO toolbox to evaluate boxes (IOU)...", logger) as block:
        logger.info("Preparing results for COCO format")
        davis_results = {}
        if "bbox" in iou_types:
            logger.info("Preparing bbox results")
            davis_results["bbox"] = prepare_for_davis_detection(predictions_coco, dataset)
        if "segm" in iou_types:
            logger.info("Preparing segm results")
            davis_results["segm"] = prepare_for_davis_segmentation(predictions_coco, dataset)

        results = DAVISResults(*iou_types)
        logger.info("Evaluating predictions")
        for iou_type in iou_types:
            with tempfile.NamedTemporaryFile() as f:
                file_path = f.name
                if output_folder:
                    file_path = os.path.join(output_folder, iou_type + ".json")
                res = evaluate_predictions_on_davis(
                    dataset.coco, davis_results[iou_type], file_path, iou_type
                )
                results.update(res)
        logger.info(results)
        check_expected_results(results, expected_results, expected_results_sigma_tol)
        if output_folder:
            torch.save(results, os.path.join(output_folder, "davis_results.pth"))
        if matching:
            # matched_prediction
            logger.info('-' * 20)
            logger.info("Preparing Evaluation on matched prediction")
            davis_results = {}
            if "bbox" in iou_types:
                logger.info("Preparing bbox results")
                davis_results["bbox"] = prepare_for_davis_detection(matched_predictions, dataset)
            if "segm" in iou_types:
                logger.info("Preparing segm results")
                davis_results["segm"] = prepare_for_davis_segmentation(matched_predictions, dataset)

            results = DAVISResults(*iou_types)
            for iou_type in iou_types:
                with tempfile.NamedTemporaryFile() as f:
                    file_path = f.name
                    if output_folder:
                        file_path = os.path.join(output_folder, iou_type + "_matched_" + ".json")
                    res = evaluate_predictions_on_davis(
                        dataset.coco, davis_results[iou_type], file_path, iou_type
                    )
                    results.update(res)
            logger.info(results)
            check_expected_results(results, expected_results, expected_results_sigma_tol)
            if output_folder:
                torch.save(results, os.path.join(output_folder, "davis_matched_results.pth"))

