import numpy as np
import torch
import torchvision.transforms as transforms



class KeepPatch(object):
    def __init__(self, size_percent=0.5, im_size=64):
        super().__init__()
        self.size_percent = size_percent

        if isinstance(size_percent, int):
            self.size_percent = (size_percent, size_percent)

        self.im_size = im_size
        self.size_patch = (int(self.size_percent[0] * im_size),
                           int(self.size_percent[1] * im_size))

    def sample_theta(self, im_shape, seed=None):
        s = np.random.RandomState(seed)

        mask = np.zeros(im_shape)
        for i in range(im_shape[0]):
            x_patch = s.randint(0, self.im_size - self.size_patch[0] + 1)
            y_patch = s.randint(0, self.im_size - self.size_patch[1] + 1)
            mask[i, :, x_patch:x_patch + self.size_patch[0], y_patch:y_patch + self.size_patch[1]] = -1

        return mask

    def measure(self, x, device, theta=None, seed=None):
        assert (self.im_size is not None)
        x_measured = x.clone()

        if theta is None:
            mask = self.sample_theta(im_shape=x.shape, seed=seed)
            mask = torch.tensor(mask, device=device, dtype=torch.bool, requires_grad=False)
            #mask = torch.tensor(mask, device=device, dtype=torch.uint8, requires_grad=False)
        else:
            mask = theta

        x_measured[mask] = 0

        return {
            "theta": mask,
            "measured_sample": x_measured,
        }


class RemovePixel(object):
    def __init__(self, percent=0.9, im_size=64):
        super().__init__()
        self.percent = percent

    def sample_theta(self, im_shape, seed=None):
        s = np.random.RandomState(seed)
        mask = np.zeros(np.prod(im_shape))
        mask[:int(self.percent * np.prod(im_shape))] = 1
        s.shuffle(mask)
        mask = mask.reshape(im_shape)
        return mask

    def measure(self, x, device, theta=None, seed=None):
        x_measured = x.clone()

        if theta is None:
            mask = self.sample_theta(im_shape=x.shape, seed=seed)
            mask = torch.tensor(mask, device=device, dtype=torch.bool, requires_grad=False)
            #mask = torch.tensor(mask, device=device, dtype=torch.uint8, requires_grad=False)
        else:
            mask = theta

        x_measured[mask] = 0
        return {
            "theta": mask,
            "measured_sample": x_measured,
        }


class RemovePixelDark(object):
    def __init__(self, percent=0.9, im_size=64):
        super().__init__()
        self.percent = percent

    def sample_theta(self, im_shape, seed=None):
        s = np.random.RandomState(seed)

        mask = np.zeros((im_shape[0], im_shape[1], im_shape[2], im_shape[3]))
        for i in range(im_shape[0]):
            ones = np.zeros([im_shape[2] * im_shape[3]])
            ones[:int(self.percent * im_shape[2] * im_shape[3])] = 1
            s.shuffle(ones)
            ones = ones.reshape(im_shape[2], im_shape[3])

            mask[i, :, ] = ones

        return mask

    def measure(self, x, device, theta=None, seed=None):
        x_measured = x.clone()

        if theta is None:
            mask = self.sample_theta(im_shape=x.shape, seed=seed)
            mask = torch.tensor(mask, device=device, dtype=torch.bool, requires_grad=False)
            #mask = torch.tensor(mask, device=device, dtype=torch.uint8, requires_grad=False)
        else:
            mask = theta

        x_measured[mask] = 0
        return {
            "theta": mask,
            "measured_sample": x_measured,
        }


class ConvNoise(object):
    def __init__(self, conv_size, noise_variance, im_size=64):
        super().__init__()
        self.conv_size = conv_size
        self.noise_variance = noise_variance

    def sample_theta(self, im_shape, seed=None):
        s = np.random.RandomState(seed)
        x = np.zeros(im_shape, dtype='f')
        x[:] = s.randn(*x.shape)

        return x * self.noise_variance

    def measure(self, x, device, theta=None, seed=None):
        x_measured = x.clone()

        if theta is None:
            noise = self.sample_theta(im_shape=x.shape, seed=seed)
            noise = torch.tensor(noise, device=device, dtype=torch.float32, requires_grad=False)
        else:
            noise = theta
        eps = torch.ones(1, 1, self.conv_size, self.conv_size, device=device) / (self.conv_size * self.conv_size)
        p = int((1 - 1 + 1 * self.conv_size) / 2)
        for i in range(x.shape[1]):
            x_measured[:, i:i + 1] = torch.nn.functional.conv2d(x[:, i:i + 1], eps, padding=p)

        return {
            "theta": noise,
            "measured_sample": (x_measured + noise),
        }

class Cloudy(object):
    def __init__(self, im_size=256):
        super().__init__()

    def sample_theta(self, im_shape, seed=None):
        s = np.random.RandomState(seed)
        x = np.zeros(im_shape, dtype='f')
        x[:] = s.randn(*x.shape)

        return x * 0.3

    def un_normalize(self, x):
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        return unnormalize(x)*255

    def measure(self, x, device, x_measured=None, theta=None, seed=None):

        #if theta is None:
           # noise = self.sample_theta(im_shape=x.shape, seed=seed)
            #noise = torch.tensor(noise, device=device, dtype=torch.float32, requires_grad=False)
        #else:
            #noise = theta
        
        if theta is None:
          torch.set_printoptions(profile="default")
          x2 = x.detach().clone()
          images = self.un_normalize(x2)
          noise = images > 150.0
          #noise = noise.float()
        else:
            noise = theta
          
            
        mask = torch.tensor(noise, device=device, dtype=torch.bool, requires_grad=False)
        if x_measured == None:
            x_measured = x.clone()
            x_measured[mask] = 0 
        return {
            "theta": noise, 
            "measured_sample": x_measured
        }