import argparse
from datetime import datetime
import time
import glob
import os
from PIL import Image
import numpy as np
import time

# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn.functional as F

# Custom includes
from src.sod_layer11 import Deeplabv3plus

# Dataloaders includes
import util.custom_transforms as trforms
from util.data_loader import SalObjDataset


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='cuda:0')
    parser.add_argument('-input_size', type=int, default=512)
    parser.add_argument('-output_stride', type=int, default=16)

    parser.add_argument('-load_path', type=str, default='')
    parser.add_argument('-save_dir', type=str, default='')

    return parser.parse_args()


def main(args):
    net = Deeplabv3plus(os = args.output_stride)
    net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.load_path,map_location="cpu").items()})
    # net.load_state_dict(torch.load(args.load_path, map_location=lambda storage, loc: storage))
    net.to(args.gpu)

    composed_transforms_ts = transforms.Compose([
        trforms.FixedResize(size=(args.input_size, args.input_size)),
        trforms.Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225)),
        trforms.ToTensor()
    ])
    
    # DUT-OMRON/  DUTS/  ECSSD/  HKU-IS/ 
    test_data = SalObjDataset(
        # image_root="datasets/DUTS/DUTS-TE/DUTS-TE-Image/",
        # gt_root="datasets/DUTS/DUTS-TE/DUTS-TE-Mask/",
        image_root = "datasets/excluded/mask_images/",
        gt_root= "datasets/excluded/mask/",
        transform=composed_transforms_ts,
        return_size=True)

    testloader = DataLoader(test_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)
    num_iter_ts = len(testloader)

    save_dir = "datasets/excluded/infer/"
    # save_dir = './SM/' + args.save_dir + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    net.eval()
    start_time = time.time()
    num_frames = 0
    with torch.no_grad():
        for i, sample_batched in enumerate(testloader):
            
            labels, label_name, size = sample_batched['label'], sample_batched['label_name'], sample_batched['size']
                    
            inputs = Variable(sample_batched['image'], requires_grad=False).to(args.gpu)

            prob_pred = net(inputs)
            prob_pred = torch.nn.Sigmoid()(prob_pred)
            prob_pred = (prob_pred - torch.min(prob_pred) + 1e-8) / (torch.max(prob_pred) - torch.min(prob_pred) + 1e-8)
            shape = (size[0, 0], size[0, 1])
            prob_pred = F.interpolate(prob_pred,
                                      size=shape,
                                      mode='bilinear',
                                      align_corners=True).cpu().data
            save_data = prob_pred[0]
            save_png = save_data[0].numpy()
            save_png = np.round(save_png * 255)
            save_png = save_png.astype(np.uint8)
            save_png = Image.fromarray(save_png)
            save_path = save_dir + label_name[0]

            if not os.path.exists(save_path[:save_path.rfind('/')]):
                os.makedirs(save_path[:save_path.rfind('/')])
            save_png.save(save_path)
            num_frames += 1
            fps = num_frames / (time.time() - start_time)
            print("progress {}/{} FPS:{}".format(i, num_iter_ts,fps), end="\r")

    duration = time.time() - start_time
    print("------------------------------------------------------------------")
    print("--%d images, cost time: %.4f s, speed: %.4f s." %
          (num_iter_ts, duration, duration / num_iter_ts))
    print("------------------------------------------------------------------")


if __name__ == '__main__':
    args = get_arguments()
    main(args)
