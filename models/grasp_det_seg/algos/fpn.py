import torch
# from inplace_abn import active_group, set_active_group

from models.grasp_det_seg.utils.bbx import shift_boxes
from models.grasp_det_seg.utils.misc import Empty
from models.grasp_det_seg.utils.parallel import PackedSequence
from models.grasp_det_seg.utils.roi_sampling import roi_sampling
from .detection import DetectionAlgo
from .rpn import RPNAlgo


class RPNAlgoFPN(RPNAlgo):
    """RPN algorithm for FPN-based region proposal networks

    Parameters
    ----------
    proposal_generator : RPNProposalGenerator
    anchor_matcher : RPNAnchorMatcher
    loss : RPNLoss
    anchor_scale : float
        Anchor scale factor, this is multiplied by the RPN stride at each level to determine the actual anchor sizes
    anchor_ratios : sequence of float
        Anchor aspect ratios
    anchor_strides: sequence of int
        Effective strides of the RPN outputs at each FPN level
    min_level : int
        First FPN level to work on
    levels : int
        Number of FPN levels to work on
    """

    def __init__(self,
                 proposal_generator,
                 anchor_matcher,
                 loss,
                 anchor_scale,
                 anchor_ratios,
                 anchor_strides,
                 min_level,
                 levels):
        super(RPNAlgoFPN, self).__init__((anchor_scale,), anchor_ratios)
        self.proposal_generator = proposal_generator
        self.anchor_matcher = anchor_matcher
        self.loss = loss
        self.min_level = min_level
        self.levels = levels

        # Cache per-cell anchors
        self.anchor_strides = anchor_strides[min_level:min_level + levels]
        self.anchors = [self._base_anchors(stride) for stride in self.anchor_strides]

    @staticmethod
    def _get_logits(head, x):
        obj_logits, bbx_logits, h, w = [], [], [], []
        for x_i in x:
            obj_logits_i, bbx_logits_i = head(x_i)
            h_i, w_i = (int(s) for s in obj_logits_i.shape[-2:])

            obj_logits_i = obj_logits_i.permute(0, 2, 3, 1).contiguous().view(obj_logits_i.size(0), -1)
            bbx_logits_i = bbx_logits_i.permute(0, 2, 3, 1).contiguous().view(bbx_logits_i.size(0), -1, 4)

            obj_logits.append(obj_logits_i)
            bbx_logits.append(bbx_logits_i)
            h.append(h_i)
            w.append(w_i)

        return torch.cat(obj_logits, dim=1), torch.cat(bbx_logits, dim=1), h, w

    def _inference(self, obj_logits, bbx_logits, anchors, valid_size, training):
        # Compute shifted boxes
        boxes = shift_boxes(anchors, bbx_logits)

        # Clip boxes to their image sizes
        for i, (height, width) in enumerate(valid_size):
            boxes[i, :, [0, 2]] = boxes[i, :, [0, 2]].clamp(min=0, max=height)
            boxes[i, :, [1, 3]] = boxes[i, :, [1, 3]].clamp(min=0, max=width)

        return self.proposal_generator(boxes, obj_logits, training)

    def training(self, head, x, bbx, iscrowd, valid_size, training=True, do_inference=False):
        # Calculate logits for the levels that we need
        x = x[self.min_level:self.min_level + self.levels]
        obj_logits, bbx_logits, h, w = self._get_logits(head, x)

        with torch.no_grad():
            # Compute anchors for each scale and merge them
            anchors = []
            for h_i, w_i, stride_i, anchors_i in zip(h, w, self.anchor_strides, self.anchors):
                anchors.append(self._shifted_anchors(
                    anchors_i, stride_i, h_i, w_i, bbx_logits.dtype, bbx_logits.device))
            anchors = torch.cat(anchors, dim=0)
            # obj_lbl: binary class label for each anchor (being an object or not)
            # bbx_lbl: coordinates for each bbx with pos object_lbl
            match = self.anchor_matcher(anchors, bbx, iscrowd, valid_size)
            obj_lbl, bbx_lbl = self._match_to_lbl(anchors, bbx, match)

        # Compute losses
        obj_loss, bbx_loss = self.loss(obj_logits, bbx_logits, obj_lbl, bbx_lbl)

        # Optionally, also run inference
        if do_inference:
            with torch.no_grad():
                proposals = self._inference(obj_logits, bbx_logits, anchors, valid_size, training)
        else:
            proposals = None

        return obj_loss, bbx_loss, proposals

    def inference(self, head, x, valid_size, training):
        # Calculate logits for the levels that we need
        x = x[self.min_level:self.min_level + self.levels]
        obj_logits, bbx_logits, h, w = self._get_logits(head, x)

        # Compute anchors for each scale and merge them
        anchors = []
        for h_i, w_i, stride_i, anchors_i in zip(h, w, self.anchor_strides, self.anchors):
            anchors.append(self._shifted_anchors(
                anchors_i, stride_i, h_i, w_i, bbx_logits.dtype, bbx_logits.device))
        anchors = torch.cat(anchors, dim=0)

        return self._inference(obj_logits, bbx_logits, anchors, valid_size, training)


class DetectionAlgoFPN(DetectionAlgo):
    """Detection algorithm for FPN networks
    """

    def __init__(self,
                 prediction_generator,
                 proposal_matcher,
                 loss,
                 classes,
                 bbx_reg_weights,
                 canonical_scale,
                 canonical_level,
                 roi_size,
                 min_level,
                 levels):
        super(DetectionAlgoFPN, self).__init__(classes, bbx_reg_weights)
        self.prediction_generator = prediction_generator
        self.proposal_matcher = proposal_matcher
        self.loss = loss
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level
        self.roi_size = roi_size
        self.min_level = min_level
        self.levels = levels

    def _target_level(self, boxes):
        scales = (boxes[:, 2:] - boxes[:, :2]).prod(dim=-1).sqrt()
        target_level = torch.floor(self.canonical_level + torch.log2(scales / self.canonical_scale + 1e-6))
        return target_level.clamp(min=self.min_level, max=self.min_level + self.levels - 1)

    def _rois(self, x, proposals, proposals_idx, img_size):
        stride = proposals.new([fs / os for fs, os in zip(x.shape[-2:], img_size)])
        proposals = (proposals - 0.5) * stride.repeat(2) + 0.5
        return roi_sampling(x, proposals, proposals_idx, self.roi_size)

    def _head(self, head, x, proposals, proposals_idx, img_size):
        # Find target levels
        target_level = self._target_level(proposals)

        # Sample rois
        rois = x[0].new_zeros(proposals.size(0), x[0].size(1), self.roi_size[0], self.roi_size[1])
        for level_i, x_i in enumerate(x):
            idx = target_level == (level_i + self.min_level)
            if idx.any().item():
                rois[idx] = self._rois(x_i, proposals[idx], proposals_idx[idx], img_size)

        # Run head
        return head(rois)

    def training(self, head, x, proposals, bbx, cat, iscrowd, img_size):
        x = x[self.min_level:self.min_level + self.levels]

        try:
            if proposals.all_none:
                raise Empty

            with torch.no_grad():
                # Match proposals to ground truth
                proposals, match = self.proposal_matcher(proposals, bbx, cat, iscrowd)
                cls_lbl, bbx_lbl = self._match_to_lbl(proposals, bbx, cat, match)

            if proposals.all_none:
                raise Empty

            # Run head
            # set_active_group(head, active_group(True))
            proposals, proposals_idx = proposals.contiguous
            cls_logits, bbx_logits = self._head(head, x, proposals, proposals_idx, img_size)

            # Calculate loss
            cls_loss, bbx_loss = self.loss(cls_logits, bbx_logits, cls_lbl, bbx_lbl)
        except Empty:
            # active_group(False)
            cls_loss = bbx_loss = sum(x_i.sum() for x_i in x) * 0

        return cls_loss, bbx_loss

    def inference(self, head, x, proposals, valid_size, img_size):
        x = x[self.min_level:self.min_level + self.levels]

        if not proposals.all_none:
            # Run head on the given proposals
            # set_active_group(head, active_group(True))
            proposals, proposals_idx = proposals.contiguous
            cls_logits, bbx_logits = self._head(head, x, proposals, proposals_idx, img_size)
            
            # Shift the proposals according to the logits
            
            bbx_reg_weights = x[0].new(self.bbx_reg_weights)
            boxes = shift_boxes(proposals.unsqueeze(1), bbx_logits / bbx_reg_weights)
            scores = torch.softmax(cls_logits, dim=1)
            

            # Split boxes and scores by image, clip to valid size
            
            boxes, scores = self._split_and_clip(boxes, scores, proposals_idx, valid_size)
            bbx_pred, cls_pred, obj_pred = self.prediction_generator(boxes, scores)
        else:
            # active_group(False)
            bbx_pred = PackedSequence([None for _ in range(x[0].size(0))])
            cls_pred = PackedSequence([None for _ in range(x[0].size(0))])
            obj_pred = PackedSequence([None for _ in range(x[0].size(0))])

        return bbx_pred, cls_pred, obj_pred