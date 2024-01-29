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
     
class swift_eye_temporal_fusion_component(nn.Module):
    def __init__(self,cfg_path) -> None:
        super().__init__()
        self.cfg_path=cfg_path
        self.cfg=Config.fromfile(cfg_path)
        self.tracking_head=MODELS.build(self.cfg.tracking_head)
        self.backbone=MODELS.build(self.cfg.backbone)
        self.neck=MODELS.build(self.cfg.neck)
        self.correlation_head=MODELS.build(self.cfg.correlation_head)
        for param in self.backbone.parameters():
            param.requires_grad=False
        for param in self.neck.parameters():
            param.requires_grad=False
        self.test_pipeline=Compose(self.cfg.test.pipeline)
        self.device=next(self.parameters()).device
        self.search_shape=33
        self.template_shape=13
        self.img_metas=[{
                            'ori_shape': (132, 132, 3),
                            'img_shape': (132, 132, 3),
                            'pad_shape': (132, 132, 3),
                            'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),}
                            ]# coresponed to the 33*4
        self.feat_w=88
        self.feat_h=72

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

    def forward(self,search,kernel,img_metas,gt_bboxes,gt_labels):
        x=self.correlation_head(kernel,search)
        feats=[]
        feats.append(x)
        feats=tuple(feats)
        losses = self.tracking_head.forward_train(feats, img_metas, gt_bboxes,
                                              gt_labels, None)
        loss, log_vars=self._parse_losses(losses)
        return loss,log_vars
    
    def get_top_left(self,center,is_search=True):#  top_left也取
        """_summary_

        Args:
            center (tensor): shape:2,the center of the bbox
            is_search (bool, optional): search_image or template image. Defaults to True.

        Returns:
            tensor(int): the top_left of roi_region
        """        
        if is_search:
            feature_shape=self.search_shape
        else:
            feature_shape=self.template_shape
        center_x=torch.floor(center[0]/4)
        if center_x>=feature_shape//2 and center_x+feature_shape//2<self.feat_w:
            top_left_x=center_x-feature_shape//2
        elif center_x<feature_shape//2:
            top_left_x=0
        else:
            top_left_x=self.feat_w-feature_shape

        center_y=torch.floor(center[1]/4)
        if center_y>=feature_shape//2 and center_y+feature_shape//2<self.feat_h:
            top_left_y=center_y-feature_shape//2
        elif center_y<feature_shape//2:
            top_left_y=0
        else:
            top_left_y=self.feat_h-feature_shape
        return torch.tensor([top_left_x,top_left_y]).int()


    def poly2center(self,poly):
        #poly: (1,8)
        obb=poly2obb_le90(poly).reshape(-1)
        center=obb[:2]
        return center


    def test(self,template_image_path,searc_image_path,template_poly,search_poly):
        datas=[]
        template_data=dict(img_info=dict(filename=template_image_path),
                   img_prefix=None)
        template_data=self.test_pipeline(template_data)
        search_data=dict(img_info=dict(filename=searc_image_path),
                     img_prefix=None)
        search_data=self.test_pipeline(search_data)
        datas.append(template_data)
        datas.append(search_data)

        data = collate(datas, samples_per_gpu=2)
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]#list(length 1) of a list(length is 4) of dict(filename,ori_shape,img_shape,flip,scale_factor)
        data['img'] = [img.data[0] for img in data['img']]
        data = scatter(data, [0])[0]
        img_as_tensor=data['img'][0]
        features=self.get_features(img_as_tensor)[0]
        template_feature=features[0]
        search_feature=features[1]

        template_poly=torch.tensor(template_poly).reshape(-1,8).to(self.device)
        search_poly=torch.tensor(search_poly).reshape(-1,8).to(self.device)
        template_center=self.poly2center(template_poly)
        search_center=self.poly2center(search_poly)
        template_top_left=self.get_top_left(template_center,is_search=False)
        search_top_left=self.get_top_left(search_center,is_search=True)
        search_feature=search_feature[:,search_top_left[1]:search_top_left[1]+self.search_shape,search_top_left[0]:search_top_left[0]+self.search_shape]
        template_feature=template_feature[:,template_top_left[1]:template_top_left[1]+self.template_shape,template_top_left[0]:template_top_left[0]+self.template_shape]
        bbox_results=self.simple_test(search_feature,template_feature)
        if len(bbox_results[0][0])>0:
            result=bbox_results[0][0][0][:5]
            result[0]+=search_top_left[0]*4
            result[1]+=search_top_left[1]*4
            return result
        else:
            return None




    def simple_test(self, search,kernel,rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        search=search.unsqueeze(0)
        kernel=kernel.unsqueeze(0)

        x=self.correlation_head(kernel,search)
        feats=[]
        feats.append(x)
        feats=tuple(feats)

        outs = self.tracking_head(feats)
        bbox_list = self.tracking_head.get_bboxes(
            *outs, self.img_metas, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.tracking_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

