import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule
import torch.nn as nn
from mmcv.runner import BaseModule
import torch


from ..builder import ROTATED_DETECTORS
@ROTATED_DETECTORS.register_module()

class CorrelationHead(BaseModule):
    """Correlation head module.

    This module is proposed in
    "SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks.
    `SiamRPN++ <https://arxiv.org/abs/1812.11703>`_.

    Args:
        in_channels (int): Input channels.
        mid_channels (int): Middle channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size of convs. Defaults to 3.
        norm_cfg (dict): Configuration of normlization method after each conv.
            Defaults to dict(type='BN').
        act_cfg (dict): Configuration of activation method after each conv.
            Defaults to dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 kernel_size=3,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None,
                 ):
        super(CorrelationHead, self).__init__(init_cfg)

        self.kernel_convs= ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.search_convs = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)


    def depthwise_correlation(self, x, kernel):
        """Depthwise cross correlation.

        This function is proposed in
        `SiamRPN++ <https://arxiv.org/abs/1812.11703>`_.

        Args:
            x (Tensor): of shape (N, C, H_x, W_x).
            kernel (Tensor): of shape (N, C, H_k, W_k).

        Returns:
            Tensor: of shape (N, C, H_o, W_o). H_o = H_x - H_k + 1. So does W_o.
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch * channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out#1,256,25,25


    def forward(self, kernel, search):
        kernel = self.kernel_convs(kernel)#1,256,5,5
        search = self.search_convs(search)#1,256,29,29
        correlation_maps = self.depthwise_correlation(search, kernel)#1,256,25,25
        return correlation_maps
    
