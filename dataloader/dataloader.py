from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os
import pandas as pd

from utils import utils
from utils.file_io import read_img, read_disp


class StereoDataset(Dataset):
    def __init__(self, data_dir,
                 dataset_name='SceneFlow',
                 mode='train',
                 save_filename=False,
                 load_pseudo_gt=False,
                 transform=None):
        super(StereoDataset, self).__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform

        sceneflow_finalpass_dict = {
            'train': 'filenames/SceneFlow_finalpass_train.txt',
            'val': 'filenames/SceneFlow_finalpass_val.txt',
            'test': 'filenames/SceneFlow_finalpass_test.txt'
        }

        kitti_2012_dict = {
            'train': 'filenames/KITTI_2012_train.txt',
            'train_all': 'filenames/KITTI_2012_train_all.txt',
            'val': 'filenames/KITTI_2012_val.txt',
            'test': 'filenames/KITTI_2012_test.txt'
        }

        kitti_2015_dict = {
            'train': 'filenames/KITTI_2015_train.txt',
            'train_all': 'filenames/KITTI_2015_train_all.txt',
            'val': 'filenames/KITTI_2015_val.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        kitti_mix_dict = {
            'train': 'filenames/KITTI_mix.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        custom = {
            'train': 'filenames/custom_train.txt',
            'val': 'filenames/custom_val.txt'
        }

        custom_full = {
            'train': 'filenames/custom_train_full.txt',
            'val': 'filenames/custom_val_full.txt'
        }

        test_sim = {
            'train': 'filenames/custom_test_sim.txt',
            'test': 'filenames/custom_test_sim.txt',
        }

        test_real = {
            'train': 'filenames/custom_test_real.txt',
            'test': 'filenames/custom_test_real.txt',
        }

        dataset_name_dict = {
            'SceneFlow': sceneflow_finalpass_dict,
            'KITTI2012': kitti_2012_dict,
            'KITTI2015': kitti_2015_dict,
            'KITTI_mix': kitti_mix_dict,
            'custom_dataset' : custom,
            'custom_dataset_full': custom_full,
            'custom_dataset_sim': test_sim,
            'custom_dataset_real': test_real,
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name

        self.samples = []

        data_filenames = dataset_name_dict[dataset_name][mode]

        lines = utils.read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()

            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]


            sample = dict()

            if self.save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]

            sample['left'] = os.path.join(data_dir, left_img)
            sample['right'] = os.path.join(data_dir, right_img)
            sample['disp'] = os.path.join(data_dir, gt_disp) if gt_disp is not None else None

            if(self.dataset_name == 'custom_dataset_full' or
                self.dataset_name == 'custom_dataset_sim' or
                self.dataset_name == 'custom_dataset_real'):
                meta = None if len(splits)<3 else splits[3]
                sample['meta'] = os.path.join(data_dir, meta) # new

                if (self.dataset_name == 'custom_dataset_sim' or
                self.dataset_name == 'custom_dataset_real'):
                    sample['label'] = os.path.join(data_dir, label) # label image

            if load_pseudo_gt and sample['disp'] is not None:
                # KITTI 2015
                if 'disp_occ_0' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ_0',
                                                                     'disp_occ_0_pseudo_gt')
                # KITTI 2012
                elif 'disp_occ' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ',
                                                                     'disp_occ_pseudo_gt')
                else:
                    raise NotImplementedError
            else:
                sample['pseudo_disp'] = None

            self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])

        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]
        if sample_path['pseudo_disp'] is not None:
            sample['pseudo_disp'] = read_disp(sample_path['pseudo_disp'], subset=subset)  # [H, W]

        if self.transform is not None:
            sample = self.transform(sample)

        if(self.dataset_name == 'custom_dataset_full' or
            self.dataset_name == 'custom_dataset_sim' or
            self.dataset_name == 'custom_dataset_real'):
            temp = pd.read_pickle(sample_path['meta'])
            sample['intrinsic'] = temp['intrinsic']
            sample['baseline'] = abs((temp['extrinsic_l']-temp['extrinsic_r'])[0][3

            if (self.dataset_name == 'custom_dataset_sim' or
            self.dataset_name == 'custom_dataset_real'):
                sample['label'] = read_img(sample_path['label']) # label image

        return sample

    def __len__(self):
        return len(self.samples)
