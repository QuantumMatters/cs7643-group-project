import torch.nn as nn

import unir.module.losses as losses
from .closure import *


class UnsupervisedImageReconstruction(Closure):
    def __init__(self, mods: dict, optims: dict, device: str, backloss_measurements: float = 0., measurement=None,
                 dict_scheduler=None, k=1, debug=False):

        assert (measurement is not None)
        self.measurement = measurement
        self.netG = mods['gen']
        self.netD = mods['dis']
        self.netG_optim = optims['gen']
        self.netD_optim = optims['dis']
        self.backloss_measurements = backloss_measurements
        self.likelihood_loss = nn.MSELoss()
        self.prior_loss = losses.GANLoss().to(device)
        self.device = device
        self.dict_scheduler = dict_scheduler
        self.lr_gen = self.dict_scheduler['gen'].get_lr()[0]
        self.lr_dis = self.dict_scheduler['dis'].get_lr()[0]
        self._register_metrics(['loss_*', 'lr_*'])
        self.k = k
        self.c = 1
        self.debug = debug

    def _forward(self, input: dict):
        self.sample = input["sample"]
        self.theta = input["theta"]
        self.measured_sample = input["measured_sample"]
        self.fake_sample = self.netG(self.measured_sample)
        measure_dict = self.measurement.measure(self.fake_sample, device=self.device)
        self.measured_fake_sample = measure_dict["measured_sample"]

        if self.backloss_measurements > 0:
            self.fake_sample_back = self.netG(self.measured_fake_sample)
            self.fake_measured_sample_back = self.measurement.measure(self.fake_sample_back, device=self.device,
                                                                      theta=measure_dict["theta"])["measured_sample"]

        self.loss_MSE = self.likelihood_loss(self.fake_sample, self.sample)
        
        return input

    def _backward_G(self):
        set_requires_grad(self.netD, False)
        self.netG_optim.zero_grad()

        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.measured_fake_sample)  # should this be measured_fake_sample?
        self.loss_G = self.prior_loss(pred_fake, True)
        if self.c % 40 == 0 and self.debug:
            print(f"pred_fake (G): {pred_fake}")
            print(f"loss_G (G): {self.loss_G}")

        if self.backloss_measurements > 0:
            loss_back_measurements = self.likelihood_loss(self.fake_measured_sample_back,
                                                          self.measured_fake_sample.detach())
            self.loss_G = self.loss_G + self.backloss_measurements * loss_back_measurements

        self.loss_G.backward()
        self.netG_optim.step()
        return self.loss_G

    def _step(self):
        self.lr_gen = self.dict_scheduler['gen'].get_lr()[0]
        self.lr_dis = self.dict_scheduler['dis'].get_lr()[0]
        self.dict_scheduler['gen'].step()
        self.dict_scheduler['dis'].step()

    def _backward_D(self):

        set_requires_grad(self.netD, True)
        self.netD_optim.zero_grad()

        pred_real = self.netD(self.measured_sample)
        pred_fake = self.netD(self.measured_fake_sample.detach().contiguous())

        # Real
        loss_D_real = self.prior_loss(pred_real, True)
        # Fake
        loss_D_fake = self.prior_loss(pred_fake, False)
        # Combined loss
        self.loss_D = (loss_D_real + loss_D_fake) * 0.5
        if self.c % 40 == 0 and self.debug:
            print(f"pred_real (D): {pred_real}")
            print(f"pred_fake (D): {pred_fake}")
            print(f"loss_D_real (D): {loss_D_real}")
            print(f"loss_D_fake (D): {loss_D_fake}")
            print(f"loss_D (D): {self.loss_D}")

        # backward
        self.loss_D.backward()
        self.netD_optim.step()

        return self.loss_D

    def _backward(self):
        self.c += 1
        self._backward_G()
        self._backward_D()
