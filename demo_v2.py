import glob

import os

import time



import torch

from PIL import Image

from vizer.draw import draw_boxes



from ssd.config import cfg

from ssd.data.datasets import COCODataset, VOCDataset

import argparse

import numpy as np



from ssd.data.transforms import build_transforms

from ssd.modeling.detector import build_detection_model

from ssd.utils import mkdir

from ssd.utils.checkpoint import CheckPointer
from ssd.utils.timer import Timer





@torch.no_grad()

def run_demo(cfg, ckpt, score_threshold, images_dir, output_dir, dataset_type):

    if dataset_type == "voc":

        class_names = VOCDataset.class_names

    elif dataset_type == 'coco':

        class_names = COCODataset.class_names

    else:

        raise NotImplementedError('Not implemented now.')

    device = torch.device(cfg.MODEL.DEVICE)



    model = build_detection_model(cfg)

    model = model.to(device)

    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)

    checkpointer.load(ckpt, use_latest=ckpt is None)

    weight_file = ckpt if ckpt else checkpointer.get_checkpoint_file()

    print('Loaded weights from {}'.format(weight_file))



    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))

    mkdir(output_dir)



    cpu_device = torch.device("cpu")

    transforms = build_transforms(cfg, is_train=False)

    model.eval()
    _t = {'im_detect': Timer()}
    timer = Timer()
    timer.tic()

    inference_time_list=[]
    load_time_list = []
    
    for image_path in image_paths:

        start = time.time()


        image_name = os.path.basename(image_path)



        image = np.array(Image.open(image_path).convert("RGB"))

        height, width = image.shape[:2]
        
        
        images = transforms(image)[0].unsqueeze(0)
        load_time = time.time() - start
        load_time_list.append(1000*load_time)
        _t['im_detect'].tic()

        #start = time.time()
        
        #print('1')
        result = model(images.to(device))[0]

        #print('2')
    


        result = result.resize((width, height)).to(cpu_device).numpy()

        boxes, labels, scores = result['boxes'], result['labels'], result['scores']



        indices = scores > score_threshold

        boxes = boxes[indices]

        labels = labels[indices]

        scores = scores[indices]
        
        #inference_time = time.time() - start
        inference_time = _t['im_detect'].toc()
        #print(1000*(inference_time))
        
        inference_time_list.append(1000*inference_time)

        meters = ' | '.join(

            [

                'objects {:02d}'.format(len(boxes)),

                'load {:03d}ms'.format(round(load_time * 1000)),

                'inference {:03d}ms'.format(round(inference_time * 1000)),

                'FPS {}'.format(round(1.0 / inference_time))

            ]

        )

       # print('({:04d}/{:04d}) {}: {}'.format(i + 1, len(image_paths), image_name, meters))



        #drawn_image = draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)

        #Image.fromarray(drawn_image).save(os.path.join(output_dir, image_name))
        _t['im_detect'].clear()

    
    
    N = len(inference_time_list)//2
    total_time_list = np.array(inference_time_list) + np.array(load_time_list)
    
     
    total_time_list.sort()
    inference_time_list.sort()
    
    det_time = np.mean(total_time_list[:N])#/BATCH_SIZE
    best_det_time = np.min(total_time_list)#/BATCH_SIZE
    
    print("Total test time: %.2f s" % (timer.toc()))
    print("\nTotal detection speed: %.1f FPS" % (len(inference_time_list)/timer.toc()))
    print("\nAvg detection speed: %.1f FPS" % (1000./det_time))
    print("Best detection speed: %.1f FPS" % (1000./best_det_time))
        





def main():

    parser = argparse.ArgumentParser(description="SSD Demo.")

    parser.add_argument(

        "--config-file",

        default="",

        metavar="FILE",

        help="path to config file",

        type=str,

    )

    parser.add_argument("--ckpt", type=str, default=None, help="Trained weights.")

    parser.add_argument("--score_threshold", type=float, default=0.7)

    parser.add_argument("--images_dir", default='demo', type=str, help='Specify a image dir to do prediction.')

    parser.add_argument("--output_dir", default='demo/result', type=str, help='Specify a image dir to save predicted images.')

    parser.add_argument("--dataset_type", default="voc", type=str, help='Specify dataset type. Currently support voc and coco.')



    parser.add_argument(

        "opts",

        help="Modify config options using the command-line",

        default=None,

        nargs=argparse.REMAINDER,

    )

    args = parser.parse_args()

    print(args)



    cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)

    cfg.freeze()



    print("Loaded configuration file {}".format(args.config_file))

    with open(args.config_file, "r") as cf:

        config_str = "\n" + cf.read()

        print(config_str)

    print("Running with config:\n{}".format(cfg))



    run_demo(cfg=cfg,

             ckpt=args.ckpt,

             score_threshold=args.score_threshold,

             images_dir=args.images_dir,

             output_dir=args.output_dir,

             dataset_type=args.dataset_type)





if __name__ == '__main__':

    main()