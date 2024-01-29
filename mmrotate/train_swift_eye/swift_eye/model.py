import torch
import torch.nn as nn
from mmrotate.core.bbox.coder import DeltaXYWHAOBBoxCoder
from mmrotate.core.anchor.anchor_generator import RotatedAnchorGenerator
from mmcv.cnn import ConvModule
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
import torch.nn.functional as F
import math
from utils import hbb2xyxy,obb2hbb_le90,xyxy2hbb,obb2poly_le90
import mmcv
import numpy as np
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
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from unet import UNet
from PIL import Image
from torchvision import transforms


import pickle
HEADS = MODELS
BACKBONES=MODELS
NECKS=MODELS
loss_kwargs=dict(beta=1.0, loss_weight=1.0)

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)




         
class swift_eye(nn.Module):
    def __init__(self,cfg_path,model_path=None) -> None:

        super().__init__()
        self.cfg=Config.fromfile(cfg_path)
        self.backbone=MODELS.build(self.cfg.backbone)
        self.neck=MODELS.build(self.cfg.neck)
        self.tracking_head=MODELS.build(self.cfg.tracking_head)
        self.detection_head=MODELS.build(self.cfg.detection_head)
        self.correlation_head=MODELS.build(self.cfg.correlation_head)
        self.unet=UNet(n_channels=1, n_classes=2, bilinear=False)

        self.template=None#store the template feature to do matching 
        if model_path is not None:
            self.load_state_dict(torch.load(model_path),strict=True)
        self.test_pipeline=Compose(self.cfg.test.pipeline)
        self.last_open_extent=None
        self.feat_h=72
        self.feat_w=88
        self.search_shape=33
        self.template_shape=13
        self.last_pred_rbbox=None#tensor shape:(5,)
        self.tracking_threshold=0
        self.detection_threshold=0.75
        self.template_update_threshold=0.95
        self.mode="detection"#detection or tracking or interpolation
        self.unet_ransform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((260, 346)),
            transforms.ToTensor()
        ])
        self.xyxy=torch.tensor([[0,0,352,288]]).cuda().float()
        self.detection_img_metas=[{
                            'ori_shape': (346, 260, 3),
                            'img_shape': (346, 260, 3),
                            'pad_shape': (346, 288, 3),
                            'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),}
                            ]# coresponed to the 33*4
        self.tracking_img_metas=[{
                            'ori_shape': (132, 132, 3),
                            'img_shape': (132, 132, 3),
                            'pad_shape': (132, 132, 3),
                            'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32),}
                            ]# coresponed to the 33*4

    def reset(self):
        self.last_open_extent=None
        self.template=None
        self.last_pred_rbbox=None
        self.mode="detection"

    def get_top_left(self,center,is_search=True):
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

    def get_first_pred(self,img_path):
        datas=[]
        data=dict(img_info=dict(filename=img_path),
                   img_prefix=None)
        data=self.test_pipeline(data)
        datas.append(data)
        data = collate(datas, samples_per_gpu=1)
        data['img'] = [img.data[0] for img in data['img']]
        data = scatter(data, [0])[0]
        img_as_tensor=data['img'][0]
        features=self.backbone(img_as_tensor)
        features=self.neck(features)
        roi_features=features[0]
        roi_features=tuple([roi_features])
        results=self.detection_head_simple_test(roi_features,self.detection_img_metas)
        if len(results[0][0])>0:#pupil exists
            rbbox_center=results[0][0][0][0:2]
            rbbox_center=torch.from_numpy(rbbox_center)
            pred_mask=self.get_pred_masks(img_path)
            pred_ep=results[0][0][0][0:5]
            open_extent=self.get_open_extent(pred_mask,pred_ep,True)
            self.last_open_extent=open_extent
            self.last_pred_rbbox=torch.from_numpy(results[0][0][0][0:5])
            if open_extent>self.template_update_threshold or self.template is None:#update template
                template_center=torch.tensor([pred_ep[0],pred_ep[1]])
                template_top_left=self.get_top_left(template_center,is_search=False)
                self.template=features[0][:, :, template_top_left[1]:template_top_left[1]+self.template_shape, template_top_left[0]:template_top_left[0]+self.template_shape]
        else:
            results=None
            open_extent=0
        return results,open_extent


    @torch.no_grad()
    def predict(self,img_path):
        top_left=self.get_top_left(self.last_pred_rbbox[:2])
        datas=[]
        data=dict(img_info=dict(filename=img_path), img_prefix=None)
        data=self.test_pipeline(data)
        datas.append(data)
        data = collate(datas, samples_per_gpu=1)
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]#list(length 1) of a list(length is 4) of dict(filename,ori_shape,img_shape,flip,scale_factor)
        data['img'] = [img.data[0] for img in data['img']]
        data = scatter(data, [0])[0]
        img_as_tensor=data['img'][0]
        features=self.backbone(img_as_tensor)
        features=self.neck(features)
        roi_features=features[0][:, :, top_left[1]:top_left[1]+self.search_shape, top_left[0]:top_left[0]+self.search_shape]
        whole_features=features[0]#1,256,72,88
        pred_mask=self.get_pred_masks(img_path)
        if self.last_open_extent>=self.detection_threshold or (self.last_open_extent>self.tracking_threshold and self.last_open_extent<self.detection_threshold and self.template is None) :#上一帧睁眼程度较大时,采用目标检测,或者还没有睁眼程度足够的
            results=self.detection_head_simple_test(tuple([whole_features]),self.detection_img_metas)
            self.mode="detection"
        elif self.last_open_extent>self.tracking_threshold and self.last_open_extent<self.detection_threshold:
            results=self.tracking_head_simple_test(roi_features,self.template)
            self.mode="tracking"
        else:
            results=self.last_pred_rbbox.numpy()
            results=results.reshape(1,1,1,5)
            confidence=np.array([0]).reshape(1,1,1,1)
            results=np.concatenate((results,confidence),axis=3)
            self.mode="interpolation"

        if len(results[0][0])>0 and results[0][0][0][5]>0:
            pred_ep=results[0][0][0][0:5]
        else:
            pred_ep=self.last_pred_rbbox.numpy()
            self.mode="interpolation"
        pred_ep=torch.from_numpy(pred_ep)
        if self.mode=="tracking" :
            pred_ep[:2]=pred_ep[:2]+top_left*4
        pred_ep=pred_ep.numpy()
        open_extent=self.get_open_extent(pred_mask,pred_ep,self.mode)

        if open_extent>self.template_update_threshold :
            template_center=torch.tensor([pred_ep[0],pred_ep[1]])
            template_top_left=self.get_top_left(template_center,False)
            self.template=features[0][:, :, template_top_left[1]:template_top_left[1]+self.template_shape, template_top_left[0]:template_top_left[0]+self.template_shape]

        if open_extent>self.tracking_threshold and self.mode=='interpolation':
            y,x=np.where(pred_mask==True)
            center_x,center_y=np.mean(x),np.mean(y)
            self.last_pred_rbbox[0]=center_x# 更新中心位置
            self.last_pred_rbbox[1]=center_y
            top_left=self.get_top_left(self.last_pred_rbbox[:2])
            roi_features=features[0][:, :, top_left[1]:top_left[1]+self.search_shape, top_left[0]:top_left[0]+self.search_shape]
        

        if (open_extent>self.tracking_threshold and open_extent<self.detection_threshold and self.mode=="interpolation") or\
        (open_extent<self.detection_threshold and open_extent>self.tracking_threshold and self.mode=="detection"):# wheter we sholud use tracking mode
            self.mode="tracking"
            results=self.tracking_head_simple_test(roi_features,self.template)
            if len(results[0][0])>0 and results[0][0][0][5]>0:# pupil exists
                pred_ep=results[0][0][0][0:5]#update
                self.mode="tracking"
            else:
                pred_ep=self.last_pred_rbbox.numpy()#pupil does not exist
                self.mode="interpolation"
            pred_ep=torch.from_numpy(pred_ep)
            if self.mode=="tracking":
                pred_ep[:2]=pred_ep[:2]+top_left*4
            pred_ep=pred_ep.numpy()
            open_extent=self.get_open_extent(pred_mask,pred_ep,self.mode)

        if open_extent>=self.detection_threshold and self.mode!="detection":# wheter we sholud use detecting mode
            self.mode="detection"
            results=self.detection_head_simple_test(tuple([whole_features]),self.detection_img_metas)
            if len(results[0][0])>0 and results[0][0][0][5]>0.5:#pupil exists
                pred_ep=results[0][0][0][0:5]#update
                self.mode="detection"
            else:
                pred_ep=self.last_pred_rbbox.numpy()#pupil does not exist
                self.mode="interpolation"
            open_extent=self.get_open_extent(pred_mask,pred_ep,self.mode)
        self.last_open_extent=open_extent
        self.last_pred_rbbox=torch.from_numpy(pred_ep)
        return pred_ep,open_extent,self.mode
    @torch.no_grad()
    def get_pred_masks_origin(self,features,xyxy):  
        """_summary_
            features:N,256,33,33
        Args:
        Returns:
            pred_boxes: batch_size,squence_length,5
        """        
        batch_size,C,H,W=features.shape
        features=self.mask_head(features)#batch_size,1,144,176
        det_labels=torch.zeros(batch_size,dtype=torch.long).cuda()
        rcnn_test_cfg = mmcv.Config({'mask_thr_binary': 0.5})
        ori_shape=(260,346)
        scale_factor=torch.tensor([1.0, 1.0, 1.0, 1.0]).cuda()
        rescale=True
        encoded_masks=self.mask_head.get_seg_masks(features,xyxy,det_labels,rcnn_test_cfg,ori_shape,scale_factor,rescale)    
        return encoded_masks[0][0]
    @torch.no_grad()
    def get_pred_masks(self,image_path):
        img = Image.open(image_path).convert('L')
        img = torch.from_numpy(np.asarray(img)[np.newaxis, ...])
        img = torch.div(img.type(torch.FloatTensor),255)
        img = img.unsqueeze(0)
        img = img.cuda()
        output = self.unet(img)
        probs = F.softmax(output, dim=1)[0]
        full_mask = self.unet_ransform(probs.cpu()).squeeze()
        full_mask=F.one_hot(full_mask.argmax(dim=0), 2).permute(2, 0, 1).numpy()
        mask=Image.fromarray((np.argmax(full_mask, axis=0) * 255 / full_mask.shape[0]).astype(np.uint8))
        result_segmentation=np.array(mask)
        y,x=np.where(result_segmentation!=0)
        result_segmentation[y,x]=1
        return result_segmentation
    def get_open_extent(self,pred_mask,pred_ep,mode):
        """ for getting the open extent of pupil

        Args:
            pred_mask (nd_array): 260,346
            pred_ep (nd_array): 5
            normal (bool, optional):tracking/detecting mode(True)interpolation mode(False)

        Returns:
            open_extent: float
        """        
        try:
            mask=np.zeros_like(pred_mask).astype(np.uint8)
            mask=cv2.ellipse(mask,(round(pred_ep[0]),round(pred_ep[1])),(round(pred_ep[2]/2),round(pred_ep[3]/2)),pred_ep[4]*180/np.pi,0,360,255,-1)
            detection_y,detection_x=np.where(mask==255)
            if mode!="interpolation":
                intersection_y,intersection_x=np.where((mask==255)&(pred_mask==True))
                open_extent=len(intersection_y)/len(detection_y)
            else:
                seg_y,seg_x=np.where(pred_mask==True)
                open_extent=len(seg_y)/len(detection_y)
            return open_extent
        except(Exception):
            print('error in get_open_extent')
    




    @torch.no_grad()
    def detection_head_simple_test(self, x,img_metas,rescale=False):
        """Test function without test time augmentation.

        Args:

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes. \
                The outer list corresponds to each image. The inner list \
                corresponds to each class.
        """
        outs = self.detection_head(x)
        bbox_list = self.detection_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.detection_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
    
    @torch.no_grad()
    def tracking_head_simple_test(self, search,kernel,rescale=False):
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


        x=self.correlation_head(kernel,search)
        feats=[]
        feats.append(x)
        feats=tuple(feats)

        outs = self.tracking_head(feats)
        bbox_list = self.tracking_head.get_bboxes(
            *outs, self.tracking_img_metas, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.tracking_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results

    def forward(self):
        pass


