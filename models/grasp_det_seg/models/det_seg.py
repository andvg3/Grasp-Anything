from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from models.grasp_det_seg.utils.sequence import pad_packed_images
from inference.models.grasp_model import GraspModel
from utils.dataset_processing.grasp import Grasp
from utils.dataset_processing.grasp import GraspRectangles

def _gr_text_to_no(l, offset=(0, 0)):
    """
    Transform a single point from a Cornell file line to a pair of ints.
    :param l: Line from Cornell grasp file (str)
    :param offset: Offset to apply to point positions
    :return: Point [y, x]
    """
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]

NETWORK_INPUTS = ["img", "msk", "bbx"]

class DetSegNet(GraspModel):
    def __init__(self,
                 body,
                 rpn_head,
                 roi_head,
                 sem_head,
                 rpn_algo,
                 detection_algo,
                 semantic_seg_algo,
                 classes):
        super(DetSegNet, self).__init__()
        self.num_stuff = classes["stuff"]

        # Modules
        self.body = body
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.sem_head = sem_head

        # Algorithms
        self.rpn_algo = rpn_algo
        self.detection_algo = detection_algo
        self.semantic_seg_algo = semantic_seg_algo

    def _prepare_inputs(self, msk, cat, iscrowd, bbx):
        cat_out, iscrowd_out, bbx_out, ids_out, sem_out = [], [], [], [], []
        for msk_i, cat_i, iscrowd_i, bbx_i in zip(msk, cat, iscrowd, bbx):
            msk_i = msk_i.squeeze(0)
            thing = (cat_i >= self.num_stuff) & (cat_i != 255)
            valid = thing & ~iscrowd_i

            if valid.any().item():
                cat_out.append(cat_i[valid])
                bbx_out.append(bbx_i[valid])
                ids_out.append(torch.nonzero(valid))
            else:
                cat_out.append(None)
                bbx_out.append(None)
                ids_out.append(None)

            if iscrowd_i.any().item():
                iscrowd_i = iscrowd_i & thing
                iscrowd_out.append(iscrowd_i[msk_i])
            else:
                iscrowd_out.append(None)

            sem_out.append(cat_i[msk_i])

        return cat_out, iscrowd_out, bbx_out, ids_out, sem_out

    def _convert_to_bb(self, bbx_pred, angle_pred):
        grs = []
        # bbxs = torch.cat(tuple(bbx_pred), dim=0)
        # angle_preds = torch.cat(tuple(angle_pred), dim=0)
        bbxs = bbx_pred[0]
        angle_preds = angle_pred[0]
        for i, bbx in enumerate(bbxs):
            x, y, w, h = bbx.tolist()
            theta = angle_preds[i].item()
            grs.append(Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr)
        grs = GraspRectangles(grs)
        # grs = grs.scale(scale)
        return grs
    
    def _numpy_to_torch(self, s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))


    def forward(self, img, msk=None, cat=None, iscrowd=None, bbx=None, do_loss=False, do_prediction=True):
        # Pad the input images
        output_size = img.shape[-1]
        device = img.device
        img.requires_grad_(True)
        img, valid_size = pad_packed_images(img)
        img_size = img.shape[-2:]

        # Convert ground truth to the internal format
        if do_loss:
            sem, _ = pad_packed_images(msk)
            msk, _ = pad_packed_images(msk)

        # Run network body
        x = self.body(img)

        # RPN part
        if do_loss:
            obj_loss, bbx_loss, proposals = self.rpn_algo.training(
                self.rpn_head, x, bbx, iscrowd, valid_size, training=self.training, do_inference=True)
        elif do_prediction:
            proposals = self.rpn_algo.inference(self.rpn_head, x, valid_size, self.training)
            obj_loss, bbx_loss = None, None
        else:
            obj_loss, bbx_loss, proposals = None, None, None

        # ROI part
        if do_loss:
            roi_cls_loss, roi_bbx_loss = self.detection_algo.training(
                self.roi_head, x, proposals, bbx, cat, iscrowd, img_size)
        else:
            roi_cls_loss, roi_bbx_loss = None, None
        if do_prediction:
            bbx_pred, cls_pred, obj_pred = self.detection_algo.inference(
                self.roi_head, x, proposals, valid_size, img_size)
        else:
            bbx_pred, cls_pred, obj_pred = None, None, None

        # Segmentation part
        # if do_loss:
        #     sem_loss, conf_mat, sem_pred,sem_logits,sem_logits_low_res, sem_pred_low_res, sem_feats  =\
        #         self.semantic_seg_algo.training(self.sem_head, x, sem, valid_size, img_size)
        # elif do_prediction:
        #     sem_pred,sem_feats,_ = self.semantic_seg_algo.inference(self.sem_head, x, valid_size, img_size)
        #     sem_loss, conf_mat = None, None
        # else:
        #     sem_loss, conf_mat, sem_pred, sem_feats = None, None, None, None

        grs = self._convert_to_bb(bbx_pred, angle_pred=obj_pred)
        pos_img, ang_img, width_img = grs.draw((output_size, output_size))
        pos = self._numpy_to_torch(pos_img).to(device)
        cos = self._numpy_to_torch(np.cos(2 * ang_img)).to(device)
        sin = self._numpy_to_torch(np.sin(2 * ang_img)).to(device)
        width = self._numpy_to_torch(width_img).to(device)

        pos.requires_grad_(True)
        cos.requires_grad_(True)
        sin.requires_grad_(True)
        width.requires_grad_(True)
        return pos, cos, sin, width
        # Prepare outputs
        # loss = OrderedDict([
        #     ("obj_loss", obj_loss),
        #     ("bbx_loss", bbx_loss),
        #     ("roi_cls_loss", roi_cls_loss),
        #     ("roi_bbx_loss", roi_bbx_loss),
        #     ("sem_loss", sem_loss)
        # ])
        # pred = OrderedDict([
        #     ("bbx_pred", bbx_pred),
        #     ("cls_pred", cls_pred),
        #     ("obj_pred", obj_pred),
        #     ("sem_pred", sem_pred)
        # ])
        # conf = OrderedDict([
        #     ("sem_conf", conf_mat)
        # ])
        # return loss, pred, conf
