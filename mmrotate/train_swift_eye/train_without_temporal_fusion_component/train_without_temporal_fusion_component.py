import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import gc
import sys
sys.path.append('mmrotate/train_swift_eye/train_without_temporal_fusion_component/regress_classify_datasets_code')# we need to modify the path to the absolute path where regress_classify_datasets_code is located in 

import mmcv
from mmcv.runner import load_checkpoint
import numpy as np
from mmrotate.models import build_detector
import torch
from mmcv.parallel import scatter
from regress_classify_datasets_code.sequence_dataset import gazeSequenceDataset
from regress_classify_datasets_code.sequence_dataloader import build_dataloader
import cv2 
from utils import obb2poly_le90,obb2poly_oc,poly2obb_le90,obb2hbb_le90,norm_angle,hbb2xyxy
from mmrotate.models.roi_heads.roi_extractors.rotate_single_level_roi_extractor import RotatedSingleRoIExtractor

from model import swift_eye_without_temporal_fusion_component
import argparse

def get_args_parser():
    parser=argparse.ArgumentParser('Set parameter for training')
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--save_path', default='', type=str)
    parser.add_argument('--train_df_path', default='', type=str)
    parser.add_argument('--val_df_path', default='', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--backbone_and_neck_path', default='', type=str)
    parser.add_argument('--config', default='', type=str,help='path of the config file of detection model')
    parser.add_argument('--last_epoch', default=-1, type=int)
    parser.add_argument('--scale', default=4, type=int,help='the scale of the feature map')
    parser.add_argument('--description', default='', type=str,help='description of the experiment')
    parser.add_argument('--feature_size', default=12, type=int,help='the size of the template')
    parser.add_argument('--feature_shift',default=0,type=int,help='the shift of the search feature')
    return parser

def set_args(args):
    """set the args

    Args:
        args (argparse): the args

    Returns:
        args
    """    
    args.save_path='path to work dir'#save the model and args
    args.train_df_path='detection_train.pickle'
    args.val_df_path='detection_validation.pickle'

    args.backbone_and_neck_path='mmrotate/train_swift_eye/train_backbone_and_neck/work_dir/epoch_30.pth'# the trained backbone and neck
    args.config='mmrotate/train_swift_eye/train_without_temporal_fusion_component/model_config.py'
    args.description='detection'
    args.resume=False   
    args.resume_path=''
    args.batch_size=8
    args.lr=1e-4
    args.scale=4
    return args

def write_args(args):
    """create a folder(model is saved in this folder) and write the args into the folder

    Args:
        args (_type_): _description_
    """    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path,"args.txt"), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")


args=get_args_parser().parse_args()
args=set_args(args)
write_args(args)


def prepare_traindataloader(args=args):
    datasets=gazeSequenceDataset(
                            test_mode=False,
                            df_path=args.train_df_path,
                            val=False)
    train_dataloader_default_args = dict(
                samples_per_gpu=args.batch_size,#batch_size
                workers_per_gpu=16,
                # `num_gpus` will be ignored if distributed
                num_gpus=1,
                dist=False,# whether to use distributed mode
                seed=520,
                runner_type="EpochBasedRunner",
                persistent_workers=False)

    dataloader=build_dataloader(datasets,**train_dataloader_default_args)
    return dataloader

def prepare_val_dataloader(args=args):
    datasets=gazeSequenceDataset(
                            test_mode=False,
                            df_path=args.val_df_path,   
                            val=True)
    val_dataloader_default_args = dict(
                samples_per_gpu=args.batch_size,# args.batch_size
                workers_per_gpu=16,
                # `num_gpus` will be ignored if distributed
                num_gpus=1,
                dist=False,# whether to use distributed mode
                seed=520,
                runner_type="EpochBasedRunner",
                persistent_workers=False)

    dataloader=build_dataloader(datasets,**val_dataloader_default_args)
    return dataloader

def prepare_refine_model(args=args):
    refine_model=swift_eye_without_temporal_fusion_component(args.config)
    weight=torch.load(args.backbone_and_neck_path,map_location=args.device)['state_dict']
    refine_model.load_state_dict(weight,strict=False)
    refine_model.to(args.device)
    return refine_model


def process_data(data):
        data_inference={}
        data_inference['img_metas'] = [data['img_metas'].data[0]]
        data_inference['img'] = [data['img'].data[0]]
        data_inference['gt_bboxes'] = [data['gt_bboxes'].data[0]]
        data_inference['gt_labels'] = [data['gt_labels'].data[0]]
        data_inference = scatter(data_inference ,[0])[0]
        img_tensor=data_inference['img'][0]

        features=refine_model.get_features(img_tensor)
        features=tuple([features[0]])
        required_data={}
        required_data['img_metas']=data_inference['img_metas'][0]
        required_data['gt_bboxes']=data_inference['gt_bboxes'][0]
        required_data['gt_labels']=data_inference['gt_labels'][0]
        required_data['feats']=features
        return required_data
            
    
print(args.description)
scale_factor=np.array([1.0, 1.0, 1.0,1.0])
train_dataloader=prepare_traindataloader(args=args)
val_dataloader=prepare_val_dataloader(args=args)
refine_model=prepare_refine_model(args=args)
optimizer = torch.optim.Adam(refine_model.parameters(),lr=args.lr)
min_val_loss_mean=100000
for epoch in range(args.last_epoch+1,30):
    refine_model.train()
    all_train_loss=0
    regress_loss_train=0
    classify_loss_train=0
    for data in train_dataloader:
        optimizer.zero_grad()
        required_data=process_data(data)
        head_losses,log_vars=refine_model.forward(**required_data)
        regress_loss_train+=log_vars['loss_bbox']
        classify_loss_train+=log_vars['loss_cls']
        head_losses.backward()
        optimizer.step()
        all_train_loss+=head_losses.item()

    print("epoch:{},train_loss:{},regress_loss:{},classify_loss:{}".format(epoch,all_train_loss/len(train_dataloader),regress_loss_train/len(train_dataloader),classify_loss_train/len(train_dataloader)))
    refine_model.eval()
    with torch.no_grad():
        all_val_loss=0
        regress_loss_val=0
        classify_loss_val=0
        for data in val_dataloader:
            required_data=process_data(data)
            head_losses,log_vars=refine_model.forward(**required_data)
            regress_loss_val+=log_vars['loss_bbox']
            classify_loss_val+=log_vars['loss_cls']
            all_val_loss+=head_losses.item()
    print("epoch:{},val_loss:{},regress_loss:{},classify_loss:{}".format(epoch,all_val_loss/len(val_dataloader),regress_loss_val/len(val_dataloader),classify_loss_val/len(val_dataloader)))
    
    if all_val_loss/len(val_dataloader)<min_val_loss_mean:
        min_val_loss_mean=all_val_loss/len(val_dataloader)
        torch.save(refine_model.state_dict(),os.path.join(args.save_path,'epoch_{}.pth'.format(epoch)))
