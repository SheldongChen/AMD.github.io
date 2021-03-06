# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import glob
import os.path as osp
import re

import mat4py
import logging
import pandas as pd
import torch

from projects.InterpretationReID.interpretationreid.data.datasets.bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY
__all__ = ['DukeMTMC_Interpretation']



# transfer pedestrian attribute to representation in duke_attribute.mat
attr2rep = {'coat': 'top',
            'long coat': 'top',
            'wearing boots': 'boots',
            'wearing hat': 'hat',
            'carrying backpack': 'backpack',
            'carrying bag': 'bag',
            'carrying handbag': 'handbag',
            'light color of shoes': 'shoes'}

# the list of empty pedestrian attributes in duke_attribute.mat
AttrEmpty = ['T-shirt', 'short of lower-body clothing']

# the list of ambiguous pedestrian attributes in duke_attribute.mat
AttrAmbig = ['long hair', 'opening an umbrella', 'pulling luggage',
             'upyellow', 'uppink', 'uporange', 'up mixed colors',
             'downpink', 'downpurple', 'downyellow', 'downorange', 'down mixed colors']

@DATASET_REGISTRY.register()
class DukeMTMC_Interpretation(ImageDataset):
    """DukeMTMC-reID.

    Reference:
        - Ristani et al. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. ECCVW 2016.
        - Zheng et al. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. ICCV 2017.

    URL: `<https://github.com/layumi/DukeMTMC-reID_evaluation>`_

    Dataset statistics:
        - identities: 1404 (train + query).
        - images:16522 (train) + 2228 (query) + 17661 (gallery).
        - cameras: 8.
    """
    dataset_dir = 'DukeMTMC-reID'
    dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-reID.zip'
    dataset_name = "dukemtmc"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self.duke_attribute_path = osp.join(self.dataset_dir, 'duke_attribute.mat')

        self.attribute_dict_all = self.generate_attribute_dict(self.duke_attribute_path,"duke_attribute")

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.duke_attribute_path,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(DukeMTMC_Interpretation, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            p_attribute = self.attribute_dict_all[str(pid)]

            if is_train:

                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid,p_attribute))

        return data

    def generate_attribute_dict(self,dir_path: str, dataset: str):

        mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
        mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)

        mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
        mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)

        mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
        mat_attribute = mat_attribute.drop(['image_index'], axis=1)
        self.key_attribute = list(mat_attribute.keys())


        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            # 1 or 2  ---->   -1 or 1
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[:].astype(int)) * 2 - 3

        return dict_attribute

    def name_of_attribute(self):
        if self.key_attribute:
            print(self.key_attribute)
            return self.key_attribute
        else:
            assert False