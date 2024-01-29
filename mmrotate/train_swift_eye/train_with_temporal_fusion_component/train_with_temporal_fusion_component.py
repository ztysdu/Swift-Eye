import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import gc
import sys
sys.path.append('mmrotate/train_swift_eye/train_with_temporal_fusion_component/regress_classify_datasets_code')# The path of regress_classify_datasets_code
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

from model import swift_eye_temporal_fusion_component
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
    parser.add_argument('--roi_size', default=7, type=int)
    parser.add_argument('--roi_channel', default=256, type=int)
    parser.add_argument('--scale', default=4, type=int,help='the scale of the feature map')
    parser.add_argument('--description', default='', type=str,help='description of the experiment')
    parser.add_argument('--template_size', default=12, type=int,help='the size of the template')
    parser.add_argument('--search_size', default=32, type=int,help='the size of the search image')
    parser.add_argument('--search_feature_shift',default=0,type=int,help='the shift of the search feature')
    parser.add_argument('--template_feature_shift',default=0,type=int,help='the shift of the template feature')
    return parser

def set_args(args):
    """set the args

    Args:
        args (argparse): the args

    Returns:
        args
    """    
    args.save_path='path to work dir'#save the model and args
    args.train_df_path='tracking_train_dataset.pickle'# training dataset
    args.val_df_path='tracking_validation_dataset.pickle'# validation dataset

    args.backbone_and_neck_path='mmrotate/train_swift_eye/train_backbone_and_neck/work_dir/epoch_30.pth'# the trained backbone and neck
    args.config='mmrotate/train_swift_eye/train_with_temporal_fusion_component/model_config.py'# the path of the configuration file

    args.description='temporal fusion component'
    args.roi_size=7
    args.roi_channel=256
    args.rotate_aug=False
    args.resume=False   
    args.resume_path=''
    args.batch_size=8
    args.lr=1e-4
    args.expansion=1
    args.scale=4
    args.template_size=13
    args.search_size=33
    args.search_feature_shift=6
    args.template_feature_shift=3
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
    refine_model=swift_eye_temporal_fusion_component(args.config)
    weight=torch.load(args.backbone_and_neck_path,map_location=args.device)['state_dict']
    refine_model.load_state_dict(weight,strict=False)
    refine_model.to(args.device)
    return refine_model



def get_corresponding_feature(feature,bbox_xyxy,is_template,obb,args=args):
    """_summary_

    Args:
        feature (_type_): 72,88
        bbox_xyxy (_type_): 4,
        is_template (bool): True:template;False:search
        obb (_type_): 5,
        args (_type_, optional): _description_. Defaults to args.
    """    
    if is_template:
        target_size=args.template_size
        shift=args.template_feature_shift
    else:
        target_size=args.search_size
        shift=args.search_feature_shift
    bbox_xyxy=bbox_xyxy/args.scale
    C,H,W=feature.shape
    x_min=torch.clamp(torch.floor(bbox_xyxy[0]).int(),min=0,max=W-1)
    x_max=torch.clamp(torch.ceil(bbox_xyxy[2]).int(),min=0,max=W-1)
    y_min=torch.clamp(torch.floor(bbox_xyxy[1]).int(),min=0,max=H-1)
    y_max=torch.clamp(torch.ceil(bbox_xyxy[3]).int(),min=0,max=H-1)
    w_size_now=x_max-x_min+1
    w_size_acquired=target_size-w_size_now
    if w_size_acquired<=0:
        x_min_real=x_min
        x_max_real=x_min_real+target_size-1
    else:
        left_acquired=torch.floor(w_size_acquired/2).int()
        shift_x=torch.min(left_acquired,torch.tensor(shift))
        if x_min-1<left_acquired-shift_x:
            x_min_real=0
            x_max_real=target_size-1
        else:
            right_capcity=W-1-x_max
            left_acquired_real=torch.randint(low=left_acquired-shift_x,high=torch.min(x_min-1,left_acquired+shift_x)+1,size=(1,)).int().cuda()#low>=0;high
            right_acquired_real=w_size_acquired-left_acquired_real         
            if right_capcity<right_acquired_real:
                x_max_real=W-1
                x_min_real=x_max_real-target_size+1
            else:

                x_min_real=x_min-left_acquired_real
                x_max_real=x_max+right_acquired_real
    h_size_now=y_max-y_min+1
    h_size_acquired=target_size-h_size_now
    if h_size_acquired<=0:
        y_min_real=y_min
        y_max_real=y_min_real+target_size-1
    else:
        top_acquired=torch.floor(h_size_acquired/2).int()
        shift_y=torch.min(top_acquired,torch.tensor(shift))
        if y_min-1<top_acquired-shift_y:
            y_min_real=0
            y_max_real=target_size-1
        else:
            bottom_capcity=H-1-y_max
            top_acquired_real=torch.randint(low=top_acquired-shift_y,high=torch.min(y_min-1,top_acquired+shift_y)+1,size=(1,)).int().cuda()      
            bottom_acquired_real=h_size_acquired-top_acquired_real         
            if bottom_capcity<bottom_acquired_real:
                y_max_real=H-1
                y_min_real=y_max_real-target_size+1
            else:
                y_min_real=y_min-top_acquired_real
                y_max_real=y_max+bottom_acquired_real
    feature=feature[:,y_min_real:y_max_real+1,x_min_real:x_max_real+1]
    poly=obb2poly_le90(obb.reshape(-1,5))
    poly[0,0::2]=poly[0,0::2]-x_min_real*args.scale
    poly[0,1::2]=poly[0,1::2]-y_min_real*args.scale
    obb_new=poly2obb_le90(poly).reshape(-1,5)
    return feature,obb_new




def process_data(data,args=args):
        data_inference={}
        data_inference['img_metas'] = [data['img_metas'].data[0]]#list(length 1) of a list(length is 32) of dict(filename,ori_shape,img_shape,flip,scale_factor)
        data_inference['img'] = [data['img'].data[0]]#[tensor:4,3,288,352](template+search)->template:2;256;12;12;search:2;256;32;32
        gt_boxes = data['gt_bboxes'].data[0]
        gt_boxes=torch.stack(tensors=gt_boxes).to(args.device)#4,1,5
        gt_boxes=gt_boxes.squeeze(1)#4,5
        gt_boxes_hbb=obb2hbb_le90(gt_boxes)#4,5
        gt_boxes_hbb_xyxy=hbb2xyxy(gt_boxes_hbb[...,:4])#4,4
        data_inference = scatter(data_inference ,[0])[0]
        img_tensor=data_inference['img'][0]#tensor:batch_size*2,C,H,W
        search_img_metas=[]

        for i in range(args.batch_size):
            search_img_meta=data_inference['img_metas'][0][i*2+1]
            search_img_meta['ori_shape']=(args.scale*args.search_size,args.scale*args.search_size,3)
            search_img_meta['img_shape']=(args.scale*args.search_size,args.scale*args.search_size,3)
            search_img_meta['pad_shape']=(args.scale*args.search_size,args.scale*args.search_size,3)
            search_img_metas.append(search_img_meta)
        
        gt_labels=[torch.tensor([0],dtype=torch.int64).cuda() for i in range(args.batch_size) ]

        template_feature_list=[]
        search_feature_list=[]
        gt_bboxes_obb_trasnformed_list=[]
        with torch.no_grad():
            features=refine_model.get_features(img_tensor)
            features=features[0]
            for i in range(args.batch_size):
                template_feature=features[i*2]#256,72,88
                template_xyxy=gt_boxes_hbb_xyxy[i*2]
                search_feature=features[i*2+1]
                search_xyxy=gt_boxes_hbb_xyxy[i*2+1]
                template_feature,template_obb=get_corresponding_feature(template_feature,template_xyxy,True,gt_boxes[i*2],args=args)
                search_feature,search_obb=get_corresponding_feature(search_feature,search_xyxy,False,gt_boxes[i*2+1],args=args)
                template_feature_list.append(template_feature)
                search_feature_list.append(search_feature)
                gt_bboxes_obb_trasnformed_list.append(search_obb)
            template_feature=torch.stack(tensors=template_feature_list)
            search_feature=torch.stack(tensors=search_feature_list)
            
            return template_feature,search_feature,gt_bboxes_obb_trasnformed_list,gt_labels,search_img_metas
            
    

train_dataloader=prepare_traindataloader(args=args)
val_dataloader=prepare_val_dataloader(args=args)
refine_model=prepare_refine_model(args=args)
optimizer = torch.optim.Adam(refine_model.parameters(),lr=args.lr)
min_val_loss_mean=100000
for epoch in range(30):
    refine_model.train()
    all_train_loss=0
    regress_loss_train=0
    classify_loss_train=0

    for data in train_dataloader:
        optimizer.zero_grad()
        template_feature,search_feature,gt_bboxes_obb_trasnformed_list,gt_labels,search_img_metas=process_data(data)
        head_losses,log_vars=refine_model.forward(search_feature,template_feature,search_img_metas,gt_bboxes_obb_trasnformed_list,gt_labels)
        head_losses.backward()
        optimizer.step()
        regress_loss_train+=log_vars['loss_bbox']
        classify_loss_train+=log_vars['loss_cls']
        all_train_loss+=head_losses.item()
    print("epoch:{},train_loss:{},regress_loss:{},classify_loss:{}".format(epoch,all_train_loss/len(train_dataloader),regress_loss_train/len(train_dataloader),classify_loss_train/len(train_dataloader)))

    refine_model.eval()
    with torch.no_grad():
        all_val_loss=0
        regress_loss_val=0
        classify_loss_val=0

        for data in val_dataloader:
            template_feature,search_feature,gt_bboxes_obb_trasnformed_list,gt_labels,search_img_metas=process_data(data)
            head_losses,log_vars=refine_model.forward(search_feature,template_feature,search_img_metas,gt_bboxes_obb_trasnformed_list,gt_labels)
            all_val_loss+=head_losses.item()
            regress_loss_val+=log_vars['loss_bbox']
            classify_loss_val+=log_vars['loss_cls']
    print("epoch:{},val_loss:{},regress_loss:{},classify_loss:{}".format(epoch,all_val_loss/len(val_dataloader),regress_loss_val/len(val_dataloader),classify_loss_val/len(val_dataloader)))

    if all_val_loss/len(val_dataloader)<min_val_loss_mean:
        min_val_loss_mean=all_val_loss/len(val_dataloader)
        torch.save(refine_model.state_dict(),os.path.join(args.save_path,'epoch_{}.pth'.format(epoch)))

    
