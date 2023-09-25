from typing import Tuple

import torch
import torchvision
from torch import Tensor
from torchvision.extension import _assert_has_ops

# from ..utils import _log_api_usage_once
# from ._box_convert import _box_cxcywh_to_xyxy, _box_xywh_to_xyxy, _box_xyxy_to_cxcywh, _box_xyxy_to_xywh
# from ._utils import _upcast

from torchvision.ops.boxes import box_iou

def nms(bboxes: torch.Tensor, scores: torch.Tensor, threshold: float=0.5, n_max=-1) -> torch.Tensor:
    order = torch.argsort(-scores)
    indices = torch.arange(bboxes.shape[0], device=bboxes.device)
    keep = torch.ones_like(indices, dtype=torch.bool, device=bboxes.device)
    for i in indices:
        if keep[i]:
            bbox = bboxes[order[i]]
            iou = box_iou(bbox[None,...],(bboxes[order[i + 1:]]) * keep[i + 1:][...,None])
            overlapped = torch.nonzero(iou > threshold)
            keep[overlapped + i + 1] = 0
    return order[keep]

# def nms(bbx, scores, threshold=0.5, n_max=-1):
#     """Perform non-maxima suppression

#     Select up to n_max bounding boxes from bbx, giving priorities to bounding boxes with greater scores. Each selected
#     bounding box suppresses all other not yet selected boxes that intersect it by more than the given threshold.

#     Parameters
#     ----------
#     bbx : torch.Tensor
#         A tensor of bounding boxes with shape N x 4
#     scores : torch.Tensor
#         A tensor of bounding box scores with shape N
#     threshold : float
#         The minimum iou value for a pair of bounding boxes to be considered a match
#     n_max : int
#         Maximum number of bounding boxes to select. If n_max <= 0, keep all surviving boxes

#     Returns
#     -------
#     selection : torch.Tensor
#         A tensor with the indices of the selected boxes

#     """
#     selection = _backend.nms(bbx, scores, threshold, n_max)
#     return selection.to(device=bbx.device)
