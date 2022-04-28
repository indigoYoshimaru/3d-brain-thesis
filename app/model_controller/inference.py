import sys
import torch
import torch.nn.functional as F
import os
import argparse
import copy
import nibabel as nib
import numpy as np
import math
from handlers.brats_data_reader import *

sys_dir = './3d-brain-thesis'
sys.path.append(os.path.dirname(sys_dir))
print(sys.path)
from models.segtran_modified.code.networks.segtran3d import Segtran3d, set_segtran3d_config, CONFIG
from app.utils.config import *
device = 'cpu'

def inference_patches(net, image, orig_patch_size, input_patch_size, batch_size, stride_xy, stride_z,
                      num_classes):
    C, H, W, D = image.shape
    dx, dy, dz = orig_patch_size

    # if any dimension of image is smaller than orig_patch_size, then padding it
    add_pad = False
    if H < dx:
        h_pad = dx - H
        add_pad = True
    else:
        h_pad = 0
    if W < dy:
        w_pad = dy - W
        add_pad = True
    else:
        w_pad = 0
    if D < dz:
        d_pad = dz - D
        add_pad = True
    else:
        d_pad = 0

    hl_pad, hr_pad = h_pad // 2, h_pad-h_pad // 2
    wl_pad, wr_pad = w_pad // 2, w_pad-w_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad-d_pad // 2
    if add_pad:
        image = F.pad(image, (0, 0, dl_pad, dr_pad, wl_pad, wr_pad, hl_pad, hr_pad),
                      mode='constant', value=0)

    # New image dimensions after padding
    C, H2, W2, D2 = image.shape

    sx = math.ceil((H2 - dx) / stride_xy) + 1
    sy = math.ceil((W2 - dy) / stride_xy) + 1
    sz = math.ceil((D2 - dz) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    device ='cpu'
    if torch.cuda.is_available(): 
        device = 'cuda'
    preds_soft = torch.zeros((num_classes, ) + image.shape[1:], device=device)
    cnt = torch.zeros_like(image[0])

    for x in range(0, sx):
        xs = min(stride_xy*x, H2-dx)
        yzs_batch = []
        test_patches = []

        for y in range(0, sy):
            ys = min(stride_xy * y, W2-dy)
            for z in range(0, sz):
                zs = min(stride_z * z, D2-dz)
                test_patch = image[:, xs:xs+dx, ys:ys+dy, zs:zs+dz]
                # test_patch: [1, 1, 144, 144, 80]
                test_patches.append(test_patch)
                yzs_batch.append([ys, zs])
                # When z == sz - 1, it's the last iteration.
                if len(test_patches) == batch_size or (y == sy - 1 and z == sz - 1):
                    # test_batch has a batch dimension after stack().
                    test_batch = torch.stack(test_patches, dim=0)
                    test_batch = F.interpolate(test_batch, size=input_patch_size,
                                               mode='trilinear', align_corners=False)
                    with torch.no_grad():
                        scores_raw = net(test_batch)

                    scores_raw = F.interpolate(scores_raw, size=orig_patch_size,
                                               mode='trilinear', align_corners=False)

                    probs = torch.sigmoid(scores_raw)
                    for i, (ys, zs) in enumerate(yzs_batch):
                        preds_soft[:, xs:xs+dx, ys:ys+dy, zs:zs+dz] += probs[i]
                        cnt[xs:xs+dx, ys:ys+dy, zs:zs+dz] += 1

                    test_patches = []
                    yzs_batch = []

    preds_soft = preds_soft / cnt.unsqueeze(dim=0)

    preds_soft = make_brats_pred_consistent(
        preds_soft, is_conservative=False)
    preds_hard = torch.zeros_like(preds_soft)
    preds_hard[1:] = (preds_soft[1:] >= 0.5)
    # A voxel is background if it's not ET, WT, TC.
    preds_hard[0] = (preds_hard[1:].sum(axis=0) == 0)

    if add_pad:
        # Remove padded pixels. clone() to make memory contiguous.
        preds_hard = preds_hard[:, hl_pad:hl_pad+H,
                                wl_pad:wl_pad+W, dl_pad:dl_pad+D].clone()
        preds_soft = preds_soft[:, hl_pad:hl_pad+H,
                                wl_pad:wl_pad+W, dl_pad:dl_pad+D].clone()
    return preds_hard, preds_soft


def inference_and_save(args, net, brain_path):
    path_head = brain_path.replace(brain_path.split('_')[-1], '')
    # args, net = load_model() # must init at the beginning! not every load time
    sample = load_brats_img_data(path_head)
    # convert sample to cuda
    if torch.cuda.is_available(): 
        sample = sample.to('cuda')
    preds_hard, preds_soft = inference_patches(net, sample, args.orig_patch_size, args.input_patch_size,
                                               1, args.orig_patch_size[0] // 2, args.orig_patch_size[2] // 2, args.num_classes)
    preds_hard = torch.argmax(brats_inv_map_label(preds_soft), dim=0)
    preds_hard += (preds_hard == 3).long()
    preds_hard_np = preds_hard.data.cpu().numpy()
    save_path = '{}pred.nii.gz'.format(path_head)
    nib.save(nib.Nifti1Image(preds_hard_np.astype(
        np.float32), np.eye(4)), save_path)
    return save_path
