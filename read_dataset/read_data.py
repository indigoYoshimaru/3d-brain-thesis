import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# from nibabel.testing import data_path
# example_file = os.path.join(data_path, 'example4d.nii.gz')
# img = nib.load(example_file)
file_path= '/mnt/d/2122sem1/pre_thesis/brain-reconstruction/code/read_dataset/BraTS2021_00000_flair.nii.gz'
print(file_path)
img = nib.load(file_path)
# print(img.get_data())

data = img.get_fdata()
print(data.shape)
def show_slices(slices):
	""" Function to display row of image slices """
	fig, axes = plt.subplots(1, len(slices))
	for i, slice in enumerate(slices):
		axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = data[26, :, :]
slice_1 = data[:, 30, :]
slice_2 = data[:, :, 16]
show_slices([slice_0, slice_1, slice_2])
# plt.suptitle("Center slices for EPI image")
from PIL import Image
img = Image.fromarray(data[:, 100, :], 'L')
img.save("image.jpeg")
