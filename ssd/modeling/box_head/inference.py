import torch
from ssd.utils.timer import Timer
from ssd.structures.container import Container
from ssd.utils.nms import batched_nms
from ssd.utils.nms import nms

class PostProcessor:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.width = cfg.INPUT.IMAGE_SIZE
        self.height = cfg.INPUT.IMAGE_SIZE

    def __call__(self, detections):
        

        batches_scores, batches_boxes = detections
        batchb = len(batches_boxes)
        #print("batch boxes: %d" % (batchb))
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        
        for batch_id in range(batch_size):
            scores, boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
            
            num_boxes = scores.shape[0]
            num_classes = scores.shape[1]
            #print(num_boxes)

            boxes = boxes.view(num_boxes, 1, 4).expand(num_boxes, num_classes, 4)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, num_classes).expand_as(scores)
            
            _t = {'nms': Timer()}
            _t['nms'].tic()

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            indices = torch.nonzero(scores > self.cfg.TEST.CONFIDENCE_THRESHOLD).squeeze(1)
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

            boxes[:, 0::2] *= self.width
            boxes[:, 1::2] *= self.height

            bb = len(boxes)
            #print("before boxes: %d" % (bb))
            keep = batched_nms(boxes, scores, labels, self.cfg.TEST.NMS_THRESHOLD)
            #keep = nms(boxes, scores,self.cfg.TEST.NMS_THRESHOLD)
            # keep only topk scoring predictions
            keep = keep[:self.cfg.TEST.MAX_PER_IMAGE]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            ab = len(boxes)
            #print("after boxes: %d" % (ab))
            #print(ab)

            container = Container(boxes=boxes, labels=labels, scores=scores)
            container.img_width = self.width
            container.img_height = self.height
            results.append(container)
            nms_time = _t['nms'].toc()
            #print("nms time: %.2f ms" % (1000*nms_time))
            print((1000*nms_time))
        
        return results
