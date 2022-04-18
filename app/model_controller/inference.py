import sys
import torch
import torch.nn.functional as F
import os
import argparse
import copy
import nibabel as nib
import numpy as np
import math

sys_dir = './3d-brain-thesis'
sys.path.append(os.path.dirname(sys_dir))
print(sys.path)
from models.segtran_modified.code.networks.segtran3d import Segtran3d, set_segtran3d_config, CONFIG
from app.utils.config import *
device = 'cpu'

def brats_inv_map_label(orig_probs):
    # orig_probs[0] is not used. Prob of 0 (background) is inferred from probs of WT.
    if type(orig_probs) == torch.Tensor:
        inv_probs = torch.zeros_like(orig_probs)
    else:
        inv_probs = np.zeros_like(orig_probs)

    inv_probs[0] = 1 - orig_probs[2]                 # P(0) = 1 - P(WT)
    inv_probs[3] = orig_probs[1]                     # P(3) = P(ET)
    UP = 1.5            # Slightly increase the prob of predicting 1 and 2.
    inv_probs[1] = orig_probs[3] - orig_probs[1]     # P(1) = P(TC) - P(ET)
    inv_probs[1] *= UP
    inv_probs[2] = orig_probs[2] - orig_probs[3]     # P(2) = P(WT) - P(TC)
    inv_probs[2] *= UP

    if (inv_probs < 0).sum() > 0:
        pdb.set_trace()

    return inv_probs


def make_brats_pred_consistent(preds_soft, is_conservative):
    # is_conservative: predict 0 as much as it can.
    # Predict 1 only when predicting 1 on all superclasses.
    # WT overrules TC; TC overrules ET. 1: ET, 2: WT, 3: TC.
    preds_soft2 = preds_soft.clone()
    if is_conservative:
        # If not WT or not TC, then not ET => P(ET) = min(P(ET), P(TC), P(WT))
        # # If not WT, then not TC         => P(TC) = min(P(TC), P(WT))
        preds_soft2[1] = torch.min(preds_soft[1:],  dim=0)[0]
        preds_soft2[3] = torch.min(preds_soft[2:],  dim=0)[0]
    # Predict 1, as long as predicting 1 on one of its subclasses.
    # ET overrules TC; TC overrules WT.
    else:
        # If TC then WT => P(WT) >= P(TC)
        preds_soft2[2] = torch.max(preds_soft[1:],    dim=0)[0]
        # If ET then TC => P(TC) >= P(ET)
        preds_soft2[3] = torch.max(preds_soft[[1, 3]], dim=0)[0]

    return preds_soft2


def transform(image):
    if image.ndim == 3:
        image = image.reshape((1,) + image.shape)
    return torch.from_numpy(image)


def process_input(image_mods, input_patch_size=[112, 112, 96], orig_patch_size=(112, 112, 96), stride_xy=56, stride_z=40):
    """Process """

    image_mm = np.stack(image_mods, axis=0)
    MOD, H, W, D = image_shape = image_mm.shape
    # create fake labels (temporary)
    labels = np.zeros_like(image_mods[0]).astype(np.uint8)
    tempL = np.nonzero(image_mm)
    # Find the boundary of non-zero voxels
    minx, maxx = np.min(tempL[1]), np.max(tempL[1])
    miny, maxy = np.min(tempL[2]), np.max(tempL[2])
    minz, maxz = np.min(tempL[3]), np.max(tempL[3])
    image_crop = image_mm[:, minx:maxx, miny:maxy, minz:maxz]
    nonzero_mask = (image_mm > 0)
    for m in range(MOD):
        image_mod = image_mm[m, :, :, :]
        image_mod_crop = image_crop[m, :, :, :]
        nonzero_voxels = image_mod_crop[image_mod_crop > 0]
        mean = nonzero_voxels.mean()
        std = nonzero_voxels.std()
        image_mm[m, :, :, :] = (image_mod - mean) / std

    # Set voxels back to 0 if they are 0 before normalization.
    image_mm *= nonzero_mask
    sample = transform(image_mm)

    return sample


def load_brats_data(path_head):
    """
    Load all modalities in brain_path. We visualize 1 mode, but need 4 of them to perform segmentation 
    """
    # load head

    modalities = ['flair', 't1ce', 't1', 't2']
    # read all files in head that has modalities
    img_mods = []
    for mod in modalities:
        file_name = '{}{}.nii.gz'.format(path_head, mod)
        print(file_name)
        nib_obj = nib.load(file_name)
        image = nib_obj.get_fdata()
        img_mods.append(image.astype(np.float32))
        # print(np.count_nonzero(image.astype(np.float32)))
    # process and transform
    sample = process_input(img_mods)
    del img_mods
    if torch.cuda.is_available(): 
        sample = sample.to('cuda')
    return sample


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
    sample = load_brats_data(path_head)
    # convert sample to cuda
    
    preds_hard, preds_soft = inference_patches(net, sample, args.orig_patch_size, args.input_patch_size,
                                               1, args.orig_patch_size[0] // 2, args.orig_patch_size[2] // 2, args.num_classes)
    preds_hard = torch.argmax(brats_inv_map_label(preds_soft), dim=0)
    preds_hard += (preds_hard == 3).long()
    preds_hard_np = preds_hard.data.cpu().numpy()
    save_path = '{}pred.nii.gz'.format(path_head)
    nib.save(nib.Nifti1Image(preds_hard_np.astype(
        np.float32), np.eye(4)), save_path)
    return save_path
