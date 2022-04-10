from __future__ import print_function

import glob

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

import numpy as np


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MNISTLoader(Dataset):
    def __init__(self, filename: str, is_train: bool = True, measurement=None):
        # Get the data file names
        if is_train:
            self.data = np.load("{}/training.npy".format(self.data_dir))
        else:
            self.data = np.load("{}/test.npy".format(self.data_dir))

        self.total = len(self.datafiles)
        print("USING MNIST")
        self.output_height = 64
        self.output_width = 64
        self.measurement = measurement
        self.transforms = transforms.Compose([
            transforms.Resize([self.output_height, self.output_width], 2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):

        x_real = self.datafiles[index]
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
