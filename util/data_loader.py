import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import re 

class SalObjDataset(data.Dataset):
    def __init__(self,image_root="", gt_root="", transform=None, return_size=False):
        self.return_size = return_size
       
        self.images_path = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.labels_path = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        assert len(self.images_path) == len(self.labels_path), f"Image and Label Error{len(self.images_path)},{len(self.labels_path)}"
        self.images_path = sorted(self.images_path)
        self.labels_path = sorted(self.labels_path)
        self.transform = transform

    def __getitem__(self, item):
        image_path = self.images_path[item]
        label_path = self.labels_path[item]
        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))
        assert os.path.exists(label_path), ('{} does not exist'.format(label_path))

        image = Image.open(image_path).convert('RGB')
        label = np.array(Image.open(label_path).convert('L'))
        label = label / label.max()
        label = Image.fromarray(label.astype(np.uint8))

        w, h = image.size
        size = (h,w)

        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        if self.return_size:
            sample['size'] = torch.tensor(size)
        
        label_name = os.path.basename(label_path)
        sample['label_name'] = label_name

        return sample

    def __len__(self):
        return len(self.images_path)