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


class CloudSatLoader(Dataset):
    def __init__(self, filename: str, is_train: bool = True, measurement=None):
        self.data_dir = filename
        # Get the data file names
        if is_train:
            self.datafiles_clear = glob.glob(self.data_dir + '/clear' + '/?[0-5]*.jpg')
            self.datafiles_cloudy = glob.glob(self.data_dir + '/cloudy' + '/?[0-5]*.jpg')
        else:
            train_clear = glob.glob(self.data_dir + '/clear' + '/?[0-5]*.jpg')
            train_cloudy = glob.glob(self.data_dir + '/cloudy' + '/?[0-5]*.jpg')
            all_clear = glob.glob(self.data_dir + '/clear/*.jpg')
            all_cloudy = glob.glob(self.data_dir + '/cloudy/*.jpg')

            self.datafiles_clear = list(set(all_clear) - set(train_clear))
            self.datafiles_cloudy = list(set(all_cloudy) - set(train_cloudy))

        self.total = len(self.datafiles_clear)
        print("USING CloudSat")
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
        x_real = self.transforms(x_real)
        x_measurement = x_real.unsqueeze(0)

        batch_cloudy= self.datafiles_cloudy[index]
        try:
            x_cloudy = pil_loader(batch_cloudy)
        except:
            return None
        x_cloudy = self.transforms(x_cloudy)

        meas = self.measurement.measure(x_measurement, device='cpu', x_measured=x_cloudy, seed=index) 
        dict_var = {
            'sample': x_real,
        }
        dict_var.update(meas)
        dict_var['mask'] = torch.tensor([0])
    
        #dict_var["measured_sample"] = dict_var["measured_sample"][0]
        return dict_var

    def __len__(self):
        return self.total
