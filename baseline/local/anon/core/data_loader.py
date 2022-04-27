"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import random

from munch import Munch
import numpy as np

import torch

class InputFetcher:
    def __init__(self, loader, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x1, x2, y1, y2, x3 = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x1, x2, y1, y2, x3 = next(self.iter_ref)
        return x1, x2, y1, y2, x3

    def __next__(self):
        x1, x2, y1, y2, x3 = self._fetch_inputs()
        # if self.mode == 'train':
        # x_ref, x_ref2, y_ref = self._fetch_refs()
        z_trg = torch.randn(x1.size(0), self.latent_dim)
        z_trg2 = torch.randn(x1.size(0), self.latent_dim)
        inputs = Munch(x_src=x1, y_src=y1, y_ref=y2,
                        x_ref=x2, x_ref2=x3,
                        z_trg=z_trg, z_trg2=z_trg2)
        # elif self.mode == 'val':
        #     x_ref, y_ref = self._fetch_inputs()
        #     inputs = Munch(x_src=x, y_src=y,
        #                    x_ref=x_ref, y_ref=y_ref)
        # elif self.mode == 'test':
        #     inputs = Munch(x=x, y=y)
        # else:
        #     raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})