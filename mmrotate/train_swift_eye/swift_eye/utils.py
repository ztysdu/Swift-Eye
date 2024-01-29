import os
import mmcv
import numpy as np
import torch
import numpy as np
import pandas as pd
import cv2

def obb2poly_le90(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)#N,2,4
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)#N,2,2
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()
def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    else:
        print('Not yet implemented.')

def poly2obb_le90(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'le90')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)

def obb2poly_oc(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rboxes[:, 0]
    y = rboxes[:, 1]
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    a = rboxes[:, 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)







def obb2hbb_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,0]
    """
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    hbboxes = torch.cat([center - bias, center + bias], dim=-1)
    _x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    _y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    _w = hbboxes[..., 2] - hbboxes[..., 0]
    _h = hbboxes[..., 3] - hbboxes[..., 1]
    _theta = theta.new_zeros(theta.size(0))
    obboxes1 = torch.stack([_x, _y, _w, _h, _theta], dim=-1)
    obboxes2 = torch.stack(tensors=[_x, _y, _h, _w, _theta - np.pi / 2], dim=-1)
    obboxes = torch.where((_w >= _h)[..., None], obboxes1, obboxes1)
    return obboxes


def hbb2xyxy(hbboxes):
    """Convert horizontal bounding boxes to bounding boxes.

    Args:
        hbboxes (torch.Tensor): [x_ctr,y_ctr,w,h]

    Returns:
        bboxes (torch.Tensor): [x0,y0,x1,y1]
    """
    x_ctr, y_ctr, w, h = torch.split(hbboxes, [1, 1, 1, 1], dim=-1)
    x0 = x_ctr - w / 2
    y0 = y_ctr - h / 2
    x1 = x_ctr + w / 2
    y1 = y_ctr + h / 2
    bboxes = torch.cat([x0, y0, x1, y1], dim=-1)
    return bboxes
# def testhbb2xyxy():
#     hbboxes=torch.Tensor([[[0,0,1,1],[0,0,2,2]]])
#     bboxes=hbb2xyxy(hbboxes)
#     print(bboxes)
# testhbb2xyxy()

def xyxy2hbb(bboxes):
    """Convert bounding boxes to horizontal bounding boxes.

    Args:
        bboxes (torch.Tensor): [x0,y0,x1,y1]

    Returns:
        hbboxes (torch.Tensor): [x_ctr,y_ctr,w,h]
    """
    x0, y0, x1, y1 = torch.split(bboxes, [1, 1, 1, 1], dim=-1)
    x_ctr = (x0 + x1) / 2
    y_ctr = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0
    hbboxes = torch.cat([x_ctr, y_ctr, w, h], dim=-1)
    return hbboxes

def calculate_iou(pred_obb,gt_obb):
    mask1=np.zeros((260,346),dtype=np.uint8)
    mask2=np.zeros((260,346),dtype=np.uint8)
    mask1=cv2.ellipse(mask1,(round(pred_obb[0]),round(pred_obb[1])),(round(pred_obb[2]/2),round(pred_obb[3]/2)),pred_obb[4]/np.pi*180,0,360,1,-1)
    mask2=cv2.ellipse(mask2,(round(gt_obb[0]),round(gt_obb[1])),(round(gt_obb[2]/2),round(gt_obb[3]/2)),gt_obb[4]/np.pi*180,0,360,1,-1)
    intersection_number=np.where((mask1==1) & (mask2==1))[0].shape[0]
    union_number=np.where((mask1==1) | (mask2==1))[0].shape[0]
    iou=intersection_number/union_number
    return iou