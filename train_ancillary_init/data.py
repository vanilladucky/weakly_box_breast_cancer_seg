from torch.utils.data import Dataset
import numpy as np
import nibabel as nib

class BreastTumor(Dataset):
    def __init__(self, info_list, transform=None):
        self.info_list = info_list
        self.transform = transform

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, idx):
        case = self.info_list[idx]
        seg_npz = f'/root/autodl-tmp/Kim/kits23/dataset/{case}/segmentation_pCE.nii.gz'
        img_npz = f'/root/autodl-tmp/Kim/kits23/dataset/{case}/imaging.nii.gz' 
        gt_npz = f'/root/autodl-tmp/Kim/kits23/dataset/{case}/segmentation.nii.gz' 
        case = case.split('.')[0]
        seg_data = nib.load(seg_npz).get_fdata()
        img_data = nib.load(img_npz).get_fdata()
        gt_data = nib.load(gt_npz).get_fdata()
        gt_data[gt_data==3] = 0 # In case of cyst

        img = img_data['data']
        seg = seg_data['seg']
        gt = gt_data['seg']

        sample = {'case': case,
                  'image': img,
                  'label': seg,
                  'gt': gt}
        if self.transform:
            sample = self.transform(sample)
        return sample

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.preload()

    def next(self):
        sample = self.sample
        self.preload()
        return sample

    def preload(self):
        try:
            self.sample = next(self.loader)
        except StopIteration:
            self.sample = None
            return