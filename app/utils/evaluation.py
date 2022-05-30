import nibabel as nib
import numpy as np
from medpy import metric
from handlers.brats_data_reader import load_brats_label

def calculate_accuracy(gt_file, pred_file): 
    num_classes = 4 
    gt_mask = load_brats_label(gt_file)
    pred_mask = load_brats_label(pred_file)
    print(type(gt_mask))
    # allcls_metric   = np.zeros((num_classes - 1, 4))
    allcls_metric = {}
    # allcls_pred_np  = gt_mask.data.cpu().numpy() # full class gt mask convert to numpy 
    # allcls_gt_np    = pred_mask.data.cpu().numpy() # full class pred mask convert to numpy
    class_name = ['Enhancing Tumor', 'Whole Tumor', 'Tumor Core']
    for cls in range(1, num_classes):
        pred = pred_mask[cls].astype(np.uint8)
        gt   = gt_mask[cls].astype(np.uint8)
   
        dice = metric.binary.dc(pred, gt)
        jc = 0
        hd = 0
        asd = 0
        if gt.sum() > 0:
            jc = metric.binary.jc(pred, gt)

        if pred.sum() > 0 and gt.sum() > 0:
            hd = metric.binary.hd95(pred, gt)
            asd  = metric.binary.asd(pred, gt)
            
        allcls_metric[class_name[cls-1]] = [dice, jc, hd, asd]
    print(allcls_metric)
    return allcls_metric