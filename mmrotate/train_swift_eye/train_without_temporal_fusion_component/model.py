import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch.nn as nn
from mmrotate.core.bbox.coder import DeltaXYWHAOBBoxCoder
from mmrotate.core.anchor.anchor_generator import RotatedAnchorGenerator
from mmcv.cnn import ConvModule
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
import torch.nn.functional as F
import math
from utils import hbb2xyxy,obb2hbb_le90,xyxy2hbb,obb2poly_le90,poly2obb_le90
import mmcv
import numpy as np
from mmdet.core.bbox.assigners import MaxIoUAssigner
from mmrotate.core.bbox.samplers import RRandomSampler
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models.losses import CrossEntropyLoss
import cv2
import os
import copy
from mmcv import Config
from mmcv.utils import build_from_cfg,Registry
from mmdet.core import images_to_levels, multi_apply
from mmdet.models.builder import MODELS
from collections import OrderedDict
import torch.distributed as dist
from mmrotate.core import rbbox2result
import pickle
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
import pandas as pd
     
class swift_eye_without_temporal_fusion_component(nn.Module):
    def __init__(self,cfg_path) -> None:
        super().__init__()
        self.cfg_path=cfg_path
        self.cfg=Config.fromfile(cfg_path)
        self.detection_head=MODELS.build(self.cfg.detection_head)
        self.backbone=MODELS.build(self.cfg.backbone)
        self.neck=MODELS.build(self.cfg.neck)
        for param in self.backbone.parameters():
            param.requires_grad=False
        for param in self.neck.parameters():
            param.requires_grad=False
        self.test_pipeline=Compose(self.cfg.test.pipeline)
        self.device=next(self.parameters()).device
        self.img_metas=[{
                            'ori_shape': (346, 260, 3),
                            'img_shape': (346, 260, 3),
                            'pad_shape': (346, 288, 3),
                            'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),}
                            ]# coresponed to the 33*4




    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
    
    def get_features(self, img):

        outs = self.backbone(img)
        x = self.neck(outs)
        return x

    def forward(self,feats,img_metas,gt_bboxes,gt_labels):
        losses = self.detection_head.forward_train(feats, img_metas, gt_bboxes,
                                              gt_labels, None)
        loss, log_vars=self._parse_losses(losses)
        return loss,log_vars
    
    def poly2center(self,poly):
        #poly: (1,8)
        obb=poly2obb_le90(poly).reshape(-1)
        center=obb[:2]
        return center

    def test(self,img_path):
        datas=[]
        data=dict(img_info=dict(filename=img_path),
                   img_prefix=None)
        data=self.test_pipeline(data)
        datas.append(data)

        data = collate(datas, samples_per_gpu=2)
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]#list(length 1) of a list(length is 4) of dict(filename,ori_shape,img_shape,flip,scale_factor)
        data['img'] = [img.data[0] for img in data['img']]
        data = scatter(data, [0])[0]
        img_as_tensor=data['img'][0]# 1,3,288,352
        features=self.get_features(img_as_tensor)[0]
        # feature=features[0]
        feature=tuple([features])
        results=self.simple_test(feature)
        if len(results[0][0])>0:
            result=results[0][0][0][:5]
        else:
            result=None
        return result



    def simple_test(self, x,rescale=False):
        """Test function without test time augmentation.

        Args:

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """

        outs = self.detection_head(x)
        bbox_list = self.detection_head.get_bboxes(
            *outs, self.img_metas, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.detection_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results



