# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# from model import swift_eye
from model import swift_eye
import cv2
import glob
import numpy as np
import shutil
import pandas as pd
import torch
from utils import poly2obb_le90,calculate_iou
tgt_image_folder='mmrotate/train_swift_eye/swift_eye/images_consequence'
if os.path.exists(tgt_image_folder):
    shutil.rmtree(tgt_image_folder)
os.mkdir(tgt_image_folder)
model=swift_eye(cfg_path='mmrotate/train_swift_eye/swift_eye/model_config.py',
                        model_path='mmrotate/train_swift_eye/swift_eye/model.pth')

model.cuda()
model.eval()
model.reset()
image_path_list=sorted(glob.glob(os.path.join('mmrotate/train_swift_eye/swift_eye/interpolated_frames','*.png')))# path of the frames
mask_path_list=[]
img_path_list=[]
pupil_obb_list=[]
iris_obb_list=[]
for i in range(len(image_path_list)):
    img_path=image_path_list[i]
    if i==0:
        results,open_extent=model.get_first_pred(img_path)
        pred_ep=results[0][0][0][0:5]
        mode="detection"
    else:
        pred_ep,open_extent,mode=model.predict(img_path)
    if mode!="interpolation":
        img=cv2.imread(img_path)
        img=cv2.ellipse(img,(round(pred_ep[0]),round(pred_ep[1])),(round(pred_ep[2]/2),round(pred_ep[3]/2)),pred_ep[4]/np.pi*180,0,360,(0,0,255),1)
        cv2.imwrite(os.path.join(tgt_image_folder,os.path.basename(img_path)),img)

