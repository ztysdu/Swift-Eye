
import torch
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
import os
import shutil
import cv2
from pipelines.loading import LoadImageFromFile, LoadAnnotations
from pipelines.transforms import RResize,Normalize,Pad,RRandomFlip,MultiScaleFlipAug,PolyRandomRotate
from pipelines.formatting import DefaultFormatBundle, Collect
from sequence_dataloader import build_dataloader
from pipelines.collate import collate_sequence

import mmcv
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from mmcv.parallel import collate
from functools import partial


import pandas as pd

angle_version = 'le90'

img_norm_cfg = dict(
    mean=[81.49, 81.49, 81.49], std=[38.74, 38.74, 38.74], to_rgb=True)


class Compose(object):
    def __init__(self,test_mode=False):
        if test_mode==False:
            self.transforms =[LoadImageFromFile(), LoadAnnotations(with_bbox=True),
                          RResize(img_scale=(346, 346)),
                          RRandomFlip(flip_ratio=[0.4, 0.1, 0.1],direction=['horizontal', 'vertical', 'diagonal'],version=angle_version),
                          PolyRandomRotate(rotate_ratio=0.3,mode='range',angles_range=45,version='le90'),
                          Normalize(**img_norm_cfg),
                          Pad( size_divisor=32),DefaultFormatBundle(), Collect(keys=['img', 'gt_bboxes', 'gt_labels'])
                          ]
        else:
            self.transforms =[LoadImageFromFile(), LoadAnnotations(with_bbox=True),
                          RResize(img_scale=(346, 346)),
                          Normalize(**img_norm_cfg),
                          Pad( size_divisor=32),DefaultFormatBundle(), Collect(keys=['img', 'gt_bboxes', 'gt_labels'])
                          ]

    def __call__(self, data,flip_direction=None,seq_number=0,rotate_angle=0):
        for t in self.transforms:
            if isinstance(t,RRandomFlip):
                data = t(data,flip_direction,seq_number)
            elif isinstance(t,PolyRandomRotate):
                data = t(data,rotate_angle,seq_number)
            else:
                data = t(data)
        return data


class gazeSequenceDataset(Dataset):
    """gaze dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    CLASSES = ("pupil",)
    PALETTE = [(165, 42, 42), ]

    def __init__(self,
                 version=angle_version,
                 difficulty=100,
                 test_mode=False,
                 df_path='/home/sduu1/userspace-18T-3/zty/tracking_pupil/df.pickle',
                 val=False):
        self.version = version
        self.difficulty = difficulty
        self.df=pd.read_pickle(df_path)
        self.data_infos = self.load_sequence_annotations(
            self.df)  # a list of dict
        self.test_mode = test_mode  # train(False) or test(True)
        self.val=val
        self.pipeline = Compose(test_mode=self.val)
        self.sequence_collate_fn =partial(collate_sequence,samples_per_gpu=2 )
        if not test_mode:
            self._set_group_flag()

    def __len__(self):
        """Total number of samples of data."""
        return len(self.df)

    def load_sequence_annotations(self, df):
        """load sequence annotations

        Args:
            df (dataframe): the first column is image path, the second column is poly

        Returns:
            sequence_data_infos:list of list of dict
        """        
        pair_data_infos=[]
        for i in range(len(df)):
            pair_data_infos.append([self.load_image_annotation(df['template_path'].iloc[i],df['origin_poly'].iloc[i]),
                                  self.load_image_annotation(df['search_path'].iloc[i],df['occlusion_poly'].iloc[i])])#load template and search image

        return pair_data_infos
    
    
    def load_image_annotation(self, image_path,poly):
        """load image annotation

        Args:
            image_path (str):image_path
            poly (one dim list): x1 y1 x2 y2 x3 y3 x4 y4
            label (int):0:closed,1:open,0-1:semi-closed
        """

        data_info = {}
        img_id = osp.split(image_path)[1][:-4]  # [:-4] to remove .png
        img_name = img_id + '.png'  # the name of the image,such as 0001.png
        data_info['filename'] = img_name
        data_info['ann'] = {}
        poly = np.array(poly, dtype=np.float32)
        x, y, w, h, a = poly2obb_np(poly, self.version)
        gt_bboxes = []
        gt_polygons = []          
        gt_bboxes.append([x, y, w, h, a])
        gt_polygons.append(poly)
        data_info['ann']['bboxes'] = np.array(
            gt_bboxes, dtype=np.float32)  # 1,5
        data_info['ann']['labels'] = np.array(
            [0], dtype=np.int64)  # 1,
        data_info['ann']['polygons'] = np.array(
            gt_polygons, dtype=np.float32)  # 1,8
        data_info['img_prefix']=osp.split(image_path)[0]
        return data_info
        
    def pre_pipeline(self, results):  # Attention,I don't know what it is used for
        """Prepare results dict for pipeline."""
        results['img_prefix'] = results['img_info']['img_prefix']  # maybe /rotate_learn/datasets/images
        results['bbox_fields'] = []

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.#不知道有啥用,但是必须有,因为build dataloader需要

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)


    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        sequence_data=self.prepare_train_img_pair(idx)
        sequence_data=self.sequence_collate_fn(sequence_data)
        return sequence_data

    

    def prepare_train_img_pair(self, idx):
        while True:
            pair_results=[]
            pair_info=self.data_infos[idx]
            template_info=pair_info[0]
            search_info=pair_info[1]
            flip_direction=None
            rotate_angle=0

            template_ann_info=template_info['ann']
            template_results=dict(img_info=template_info,ann_info=template_ann_info)
            self.pre_pipeline(template_results)
            template_results=self.pipeline(template_results,flip_direction,seq_number=0,rotate_angle=rotate_angle)
            if template_results is None:
                idx = (idx+60) % len(self)
                continue
            flip_direction=template_results['img_metas'].data['flip_direction']
            rotate_angle=template_results['img_metas'].data['rotate_angle']#旋转角度

            search_ann_info=search_info['ann']
            search_results=dict(img_info=search_info,ann_info=search_ann_info)
            self.pre_pipeline(search_results)
            search_results=self.pipeline(search_results,flip_direction,seq_number=1,rotate_angle=rotate_angle)
            if search_results is None:
                idx = (idx+60) % len(self)
                continue
            pair_results.append(template_results)
            pair_results.append(search_results)
            return pair_results




    
    

