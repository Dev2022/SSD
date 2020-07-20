from torch import nn
from ssd.utils.timer import Timer

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head


class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)

    def forward(self, images, targets=None):
        #print('1')
        _t = {'forward': Timer()}
        _t['forward'].tic()

        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)

        forward_time = _t['forward'].toc()
        #print("total forward time: %.4f ms" % (forward_time*1000))
        #print((forward_time*1000))
        if self.training:
            return detector_losses
        return detections
