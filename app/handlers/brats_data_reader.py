import torch
import argparse
import copy
import nibabel as nib
import numpy as np
import math

def brats_map_label(mask):
    num_classes = 4
    if type(mask) == torch.Tensor:
        mask_nhot = torch.zeros((num_classes,) + mask.shape, device='cuda')
    else:
        mask_nhot = np.zeros((num_classes,) + mask.shape)

    # 1 for NCR & NET, 2 for ED, 3 for ET, and 0 for everything else.
    mask_nhot[0, mask==0] = 1
    mask_nhot[1, mask==3] = 1                               # P(ET) = P(3)
    mask_nhot[2, (mask==3) | (mask==1) | (mask==2)] = 1 # P(WT) = P(1)+P(2)+P(3)
    mask_nhot[3, (mask==3) | (mask==1)] = 1               # P(TC) = P(1)+P(3)
    # Has the batch dimension. Swap the batch to the zero-th dim.

    if len(mask_nhot.shape) == 5:
        mask_nhot = mask_nhot.permute(1, 0, 2, 3, 4)
    return mask_nhot

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

def load_brats_img_data(path_head):
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

    return sample

def load_brats_label(file_path):
    nib_obj = nib.load(file_path)
    image = nib_obj.get_fdata()    
    image = image.astype(np.uint8)
    image -= (image==4)
    return brats_map_label(image)