import nibabel as nib
import numpy as np
import torch

filename = '/home/seito/Documents/study/IU/3d-brain-thesis/app/data/sample/BraTS2021_00024/BraTS2021_00024_seg.nii.gz'
mask = nib.load(filename)
print(mask)
data = mask.get_fdata()
print(data[data!=0])
converted = torch.from_numpy(data)
print(converted[converted>=4])
unique = torch.unique(torch.from_numpy(data))
print(unique)

count=0
for x in data: 
    count+=1
    # print(f'count: {count}')
    # print(f'x:{x}, x_shape: {x.shape}')
    # img = nib.Nifti1Image(x, np.eye(4))
    # nib.save(img, f'test{count}.nii.gz')  
count = 0
# for x in np.nonzero(data): 
#     print(f'x:{x}, x_shape: {x.shape}')
x = np.nonzero(data)
min_val = min(min(x[0]), min(x[1]), min(x[2]))
# mean_val = np.mean(np.mean(x[0]), np.mean(x[1]), np.mean(x[2]))
max_val = max(max(x[0]), max(x[1]), max(x[2]))

print(f'min: {min_val}, max: {max_val}')
# img = nib.Nifti1Image(x==1, np.eye(4))
# nib.save(img, f'test{count}.nii.gz')  


