from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import glob
import nibabel as nib


class CTDataset(Dataset):
    def __init__(self, datapath, transforms_):
        self.datapath = datapath
        self.transforms = transforms_
        self.samples = ['../..'+x.split('.')[4]
                        for x in glob.glob(self.datapath + '/*.im')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # print(self.samples[idx])
        image = h5py.File(self.samples[idx] + '.im', 'r').get('data')[()]
        mask = h5py.File(self.samples[idx] + '.seg', 'r').get('data')[()]
        # print(self.samples[idx])
        # print(image.shape)
        # print(mask.shape)
        if self.transforms:
            image, mask = self.transforms(image), self.transforms(mask)

        return {"A": image, "B": mask}


# update class for BraTS dataset

class BraTSDataset(Dataset):
    def read_images(self, img_dir: str) -> list:
        # images lists
        t1_list = sorted(
            glob.glob('{}/**/*t1.nii.gz'.format(img_dir), recursive=True))
        t2_list = sorted(
            glob.glob('{}/**/*t2.nii.gz'.format(img_dir), recursive=True))
        t1ce_list = sorted(
            glob.glob('{}/**/*t1ce.nii.gz'.format(img_dir), recursive=True))
        flair_list = sorted(
            glob.glob('{}/**/*flair.nii.gz'.format(img_dir), recursive=True))
        seg_list = sorted(glob.glob('{}/**/*seg.nii.gz'.format(img_dir)))

        Nim = len(t1_list)
        img_list = []

        for i in range(0, Nim):
            img_list.append([t1_list[i], t2_list[i], t1ce_list[i],
                             flair_list[i], seg_list[i]])
        return img_list

    def __init__(self, datapath, transforms_):
        self.datapath = datapath
        self.transforms = transforms_
        self.samples = self.read_images(datapath)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        images = []
        mask = []
        sample = self.samples[index]
        for idx in range(0, len(sample)-2):
            modal = nib.load(sample[idx]).get_fdata(
                dtype='float32', caching='unchanged')
            images.append(modal)
        images = np.asarray(images)

        mask = nib.load(sample[len(sample)-1]).get_fdata(
            dtype='float32', caching='unchanged')

        return {"A": images, "B": mask}
