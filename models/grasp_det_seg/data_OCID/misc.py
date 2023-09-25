import torch
import numpy as np

from grasp_det_seg.utils.parallel import PackedSequence


def iss_collate_fn(items):
    """Collate function for ISS batches"""
    out = {}
    if len(items) > 0:
        for key in items[0]:
            out[key] = [item[key] for item in items]
            if isinstance(items[0][key], torch.Tensor):
                out[key] = PackedSequence(out[key])
    return out

def prepare_frcnn_format(boxes,im_size):
    boxes_ary = np.asarray(boxes)

    boxes_ary = np.swapaxes(boxes_ary, 1, 2)
    xy_ctr = np.sum(boxes_ary, axis=2) / 4
    x_ctr = xy_ctr[:, 0]
    y_ctr = xy_ctr[:, 1]
    width = np.sqrt(np.sum((boxes_ary[:, :, 0] - boxes_ary[:, :, 1]) ** 2, axis=1))
    height = np.sqrt(np.sum((boxes_ary[:, :, 1] - boxes_ary[:, :, 2]) ** 2, axis=1))

    theta = np.zeros((boxes_ary.shape[0]), dtype=np.int)
    theta = np.arctan((boxes_ary[:, 1, 1] - boxes_ary[:, 1, 0]) / (boxes_ary[:, 0, 0] - boxes_ary[:, 0, 1]))
    b = np.arctan((boxes_ary[:, 1, 0] - boxes_ary[:, 1, 1]) / (boxes_ary[:, 0, 1] - boxes_ary[:, 0, 0]))
    theta[np.where(boxes_ary[:, 0, 0] <= boxes_ary[:, 0, 1])] = b[np.where(boxes_ary[:, 0, 0] <= boxes_ary[:, 0, 1])]

    # used for fasterrcnn loss
    x_min = x_ctr - width / 2
    x_max = x_ctr + width / 2
    y_min = y_ctr - height / 2
    y_max = y_ctr + height / 2

    x_coords = np.vstack((x_min, x_max))
    y_coords = np.vstack((y_min, y_max))

    mat = np.asarray((np.all(x_coords > im_size[1], axis=0), np.all(x_coords < 0, axis=0),
                      np.all(y_coords > im_size[0], axis=0), np.all(y_coords < 0, axis=0)))

    fail = np.any(mat, axis=0)
    correct_idx = np.where(fail == False)
    theta_deg = np.rad2deg(theta) + 90
    cls = (np.round((theta_deg) / (180 / 18))).astype(int)
    cls[np.where(cls == 18)] = 0

    ret_value = (boxes_ary[correct_idx], theta_deg[correct_idx],cls[correct_idx])
    return ret_value

def read_boxes_from_file(gt_path,delta_xy):
    with open(gt_path)as f:
        points_list = []
        box_list = []
        for count, line in enumerate(f):
            line = line.rstrip()
            [x, y] = line.split(' ')
            x = float(x) - int(delta_xy[0])
            y = float(y) - int(delta_xy[1])

            pt = (x, y)
            points_list.append(pt)

            if len(points_list) == 4:
                box_list.append(points_list)
                points_list = []
    return box_list
