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
    def read_images(img_dir: str) -> list:
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

    def load_img(img_files):
        """ Load one image and its target form file
        """
        N = len(img_files)
        # target
        y = nib.load(
            img_files[N-1]).get_fdata(dtype='float32', caching='unchanged')
        y = y[40:200, 34:226, 8:136]
        y[y == 4] = 3

        X_norm = np.empty((240, 240, 155, 4))
        for channel in range(N-1):
            X = nib.load(img_files[channel]).get_fdata(
                dtype='float32', caching='unchanged')
            brain = X[X != 0]
            brain_norm = np.zeros_like(X)  # background at -100
            norm = (brain - np.mean(brain))/np.std(brain)
            brain_norm[X != 0] = norm
            X_norm[:, :, :, channel] = brain_norm

        X_norm = X_norm[40:200, 34:226, 8:136, :]
        del(X, brain, brain_norm)

        return X_norm, y

    def __init__(self, datapath, transforms_):
        self.datapath = datapath
        self.transforms = transforms_
        self.samples = read_images(datapath)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        samples_temp = [self.samples[k] for k in indexes]

        # Generate data
        images, mask = self.__data_generation(samples_temp)

        if index == self.__len__()-1:
            self.on_epoch_end()

        return {"A": images, "B": mask}

    def __data_generation(self, samples_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, IDs in enumerate(samples_temp):
            # Store sample
            X[i], y[i] = load_img(IDs)

        return X.astype('float32'), y
