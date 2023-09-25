from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional
# from inplace_abn import ABN
from models.grasp_det_seg.utils.misc import try_index

class FPNROIHead(nn.Module):
    """ROI head module for FPN
    """

    def __init__(self, in_channels, classes, roi_size, hidden_channels=1024, norm_act=nn.BatchNorm2d):
        super(FPNROIHead, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(int(roi_size[0] * roi_size[1] * in_channels / 4), hidden_channels, bias=False)),
            ("bn1",nn.BatchNorm1d(hidden_channels)),
            ("fc2", nn.Linear(hidden_channels, hidden_channels, bias=False)),
            ("bn2", nn.BatchNorm1d(hidden_channels))
        ]))
        self.roi_cls = nn.Linear(hidden_channels, classes["thing"] + 1)
        self.roi_bbx = nn.Linear(hidden_channels, classes["thing"] * 4)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu", self.fc.bn1.weight)

        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear):
                if "roi_cls" in name:
                    nn.init.xavier_normal_(mod.weight, .01)
                elif "roi_bbx" in name:
                    nn.init.xavier_normal_(mod.weight, .001)
                else:
                    nn.init.xavier_normal_(mod.weight, gain)
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1.)

            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    def forward(self, x):
        """ROI head module for FPN
        """
        x = functional.avg_pool2d(x, 2)

        # Run head
        x = self.fc(x.view(x.size(0), -1))
        return self.roi_cls(x), self.roi_bbx(x).view(x.size(0), -1, 4)

class FPNSemanticHeadDeeplab(nn.Module):
    """Semantic segmentation head for FPN-style networks, extending Deeplab v3 for FPN bodies"""

    class _MiniDL(nn.Module):
        def __init__(self, in_channels, out_channels, dilation, pooling_size, norm_act):
            super(FPNSemanticHeadDeeplab._MiniDL, self).__init__()
            self.pooling_size = pooling_size

            self.conv1_3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
            self.conv1_dil = nn.Conv2d(in_channels, out_channels, 3, dilation=dilation, padding=dilation, bias=False)
            self.conv1_glb = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels * 3)

            self.conv2 = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)

        def _global_pooling(self, x):
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = functional.avg_pool2d(x, pooling_size, stride=1)
            pool = functional.pad(pool, pad=padding, mode="replicate")
            return pool

        def forward(self, x):
            x = torch.cat([
                self.conv1_3x3(x),
                self.conv1_dil(x),
                self.conv1_glb(self._global_pooling(x)),
            ], dim=1)
            x = self.bn1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            return x

    def __init__(self,
                 in_channels,
                 min_level,
                 levels,
                 num_classes,
                 hidden_channels=128,
                 dilation=6,
                 pooling_size=(64, 64),
                 norm_act=nn.BatchNorm2d,
                 interpolation="bilinear"):
        super(FPNSemanticHeadDeeplab, self).__init__()
        self.min_level = min_level
        self.levels = levels
        self.interpolation = interpolation

        self.output = nn.ModuleList([
            self._MiniDL(in_channels, hidden_channels, dilation, pooling_size, norm_act) for _ in range(levels)
        ])
        self.conv_sem = nn.Conv2d(hidden_channels * levels, num_classes, 1)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu", self.output[0].bn1.weight)
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Conv2d):
                if "conv_sem" not in name:
                    nn.init.xavier_normal_(mod.weight, gain)
                else:
                    nn.init.xavier_normal_(mod.weight, .1)
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1.)
            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    def forward(self, xs):
        xs = xs[self.min_level:self.min_level + self.levels]

        ref_size = xs[0].shape[-2:]
        interp_params = {"mode": self.interpolation}
        if self.interpolation == "bilinear":
            interp_params["align_corners"] = False

        for i, output in enumerate(self.output):
            xs[i] = output(xs[i])
            if i > 0:
                xs[i] = functional.interpolate(xs[i], size=ref_size, **interp_params)

        xs_feats = torch.cat(xs, dim=1)
        xs = self.conv_sem(xs_feats)

        return xs,xs_feats
