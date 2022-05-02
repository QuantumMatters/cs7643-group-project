from __future__ import print_function

import glob

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

import numpy as np
import torch


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB') #RGB or Grey Scale? 

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return resize(x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w], mode='constant', anti_aliasing=True)

class CloudSatLoader2(Dataset):
    def __init__(self, filename: str, is_train: bool = True, measurement=None):
        self.data_dir = filename
        # Get the data file names
        if is_train:
            self.datafiles_clear = sorted(glob.glob(self.data_dir + '/clear' + '/?[0-5]*.jpg'))

        else:
            train_clear = glob.glob(self.data_dir + '/clear' + '/?[0-5]*.jpg')
            all_clear = glob.glob(self.data_dir + '/clear/*.jpg')
            self.datafiles_clear = sorted(list(set(all_clear) - set(train_clear)))

        self.total = len(self.datafiles_clear)
        print("USING CloudSat2")
        self.output_height = 64
        self.output_width = 64
        self.measurement = measurement
        self.transforms = transforms.Compose([
            transforms.Resize([self.output_height, self.output_width], 2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    
    def __getitem__(self, index):

        batch_file = self.datafiles_clear[index]
        try:
            x_real = pil_loader(batch_file)
        except:
            return None

        cropped_image = center_crop(x_real, 256, 256, 64, 64)

        x_real = self.transforms(x_real)
        x_measurement = x_real.unsqueeze(0)

        meas = self.measurement.measure(x_measurement, device='cpu', seed=index)
        dict_var = {
            'sample': x_real,
        }
        dict_var.update(meas)
        if 'mask' in dict_var:
            dict_var['mask'] = dict_var['mask'][0]
        dict_var["measured_sample"] = dict_var["measured_sample"][0]  # because the dataloader add a dimension
        return dict_var

    def __len__(self):
        return self.total
