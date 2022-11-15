__copyright__ = \
"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 11/11/2019 
"""
__license__ = "CC BY-NC-SA 4.0"
__authors__ = "Javier Ribera, David Guera, Yuhao Chen, Edward J. Delp"
__version__ = "1.6.0"


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,
                 ultrasmall=False,
                 device=torch.device('cuda')):
        """
        Instantiate a UNet network.
        :param n_channels: Number of input channels (e.g, 3 for RGB)
        :param n_classes: Number of output classes
        :param height: Height of the input images
        :param known_n_points: If you know the number of points,
                               (e.g, one pupil), then set it.
                               Otherwise it will be estimated by a lateral NN.
                               If provided, no lateral network will be build
                               and the resulting UNet will be a FCN.
        :param ultrasmall: If True, the 5 central layers are removed,
                           resulting in a much smaller UNet.
        :param device: Which torch device to use. Default: CUDA (GPU).
        """
        super(UNet, self).__init__()

        self.ultrasmall = ultrasmall
        self.device = device
        
        self.inc = inconv(n_channels, 8)
        self.down1 = down(8, 16)
        self.down2 = down(16, 32)
        if self.ultrasmall:
            self.down3 = down(32, 64, normaliz=False)
            self.up1 = up(96, 32)
            self.up2 = up(48, 16)
            self.up3 = up(24, 8, activ=False)
        else:
            self.down3 = down(16, 16)
            self.down4 = down(16, 16)
            self.down5 = down(16, 32)
            self.down6 = down(32, 32)
            self.down7 = down(32, 32, normaliz=False)
            
            self.up1 = up(64, 32)
            self.up2 = up(64, 16)
            self.up3 = up(32, 16)
            self.up4 = up(32, 16)
            self.up5 = up(32, 8)
            self.up6 = up(16, 8)
            self.up7 = up(16, 8, activ=False)

        self.outc = outconv(8, n_classes)
        self.out_nonlin = nn.Sigmoid()
        
        # This layer is not connected anywhere
        # It is only here for backward compatibility
        self.lin = nn.Linear(1, 1, bias=False)


    def forward(self, x):

        batch_size = x.shape[0]

        x1 = self.inc(x)
        # print("x1: ", x1.shape)
        x2 = self.down1(x1)
        # print("x2: ", x2.shape)
        x3 = self.down2(x2)
        # print("x3: ", x3.shape)
        x4 = self.down3(x3)
        # print("x4: ", x4.shape)
        if self.ultrasmall:
            x = self.up1(x4, x3)
            # print("x up1: ", x.shape)
            x = self.up2(x, x2)
            # print("x up2: ", x.shape)
            x = self.up3(x, x1)
            # print("x up3: ", x.shape)
        else:
            x5 = self.down4(x4)
            # print("x5: ", x5.shape)
            x6 = self.down5(x5)
            # print("x6: ", x6.shape)
            x7 = self.down6(x6)
            # print("x7: ", x7.shape)
            x8 = self.down7(x7)
            # print("x8: ", x8.shape)
            
            x = self.up1(x8, x7)
            # print("up1: ", x.shape)
            x = self.up2(x, x6)
            # print("up2: ", x.shape)
            x = self.up3(x, x5)
            # print("up3: ", x.shape)
            x = self.up4(x, x4)
            # print("up4: ", x.shape)
            x = self.up5(x, x3)
            # print("up5: ", x.shape)
            x = self.up6(x, x2)
            # print("up6: ", x.shape)
            x = self.up7(x, x1)
            # print("up7: ", x.shape)

        x = self.outc(x)
        x = self.out_nonlin(x)
        
        return x


"""
Copyright &copyright © (c) 2019 The Board of Trustees of Purdue University and the Purdue Research Foundation.
All rights reserved.

This software is covered by US patents and copyright.
This source code is to be used for academic research purposes only, and no commercial use is allowed.

For any questions, please contact Edward J. Delp (ace@ecn.purdue.edu) at Purdue University.

Last Modified: 11/11/2019 
"""
