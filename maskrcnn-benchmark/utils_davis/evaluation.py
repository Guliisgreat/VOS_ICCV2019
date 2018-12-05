import numpy as np


def davis_toolbox_evaluation(output_dir):
    """
    Use default DAVIS toolbox to evaluate the performance (J, F)
    Print the result on the table
    Save the result into a yaml file

    Args:
        output_dir: the folder includes all final annotations
    """
    import os.path as osp
    import itertools
    import yaml

    from utils_davis.davis_toolbox import Timer, log, cfg, db_eval, phase, DAVISLoader, Segmentation
    from prettytable import PrettyTable

    phase = phase.VAL
    # phase = phase[config.dataset.test_image_set.replace('_480p', '')]

    # Load DAVIS
    db = DAVISLoader('2017', phase, False)
    print('Loading video segmentations from: {}'.format(output_dir))
    # Load segmentation
    segmentations = [Segmentation(
        osp.join(output_dir, s), False) for s in db.iternames()]
    # Evaluate results
    evaluation = db_eval(db, segmentations, ['J', 'F'])
    # Print results
    table = PrettyTable(['Method'] + [p[0] + '_' + p[1] for p in
                                      itertools.product(['J', 'F'], ['mean', 'recall', 'decay'])])
    table.add_row([osp.basename(output_dir)] + ["%.3f" % np.round(
        evaluation['dataset'][metric][statistic], 3) for metric, statistic
                                                in itertools.product(['J', 'F'], ['mean', 'recall', 'decay'])])
    print(str(table) + "\n")
    # Save results into yaml file
    with open(osp.join(output_dir, 'davis_eval_results.yaml'), 'w') as f:
        yaml.dump(evaluation, f)




def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box))
    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box))

    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou
