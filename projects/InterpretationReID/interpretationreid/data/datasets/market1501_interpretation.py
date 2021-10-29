# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings
import mat4py
import logging
import pandas as pd
import torch
from projects.InterpretationReID.interpretationreid.data.datasets.bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY

__all__ = ['Market1501_Interpretation']


# transfer pedestrian attribute to representation in market_attribute.mat
attr2rep = {'long hair': 'hair',
            'T-shirt': 'up',
            'coat': 'up',
            'short of lower-body clothing': 'down',
            'type of lower-body clothing (pants)': 'clothes',
            'wearing hat': 'hat',
            'carrying backpack': 'backpack',
            'carrying bag': 'bag',
            'carrying handbag': 'handbag'}

# the list of empty pedestrian attributes in market_attribute.mat
AttrEmpty = ['wearing boots', 'long coat']

# the list of ambiguous pedestrian attributes in market_attribute.mat
AttrAmbig = ['light color of shoes', 'opening an umbrella', 'pulling luggage',
             'upbrown', 'uppink', 'uporange', 'up mixed colors',
             'downred', 'downorange', 'down mixed colors']

@DATASET_REGISTRY.register()
class Market1501_Interpretation(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'Market-1501-v15.09.15'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    dataset_name = "market1501"

    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        self.logger = logging.getLogger('fastreid.' + __name__)
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market1501-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market1501-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        self.market_attribute_path = osp.join(self.data_dir, 'market_attribute.mat')
        self.attribute_dict_all = self.generate_attribute_dict(self.market_attribute_path,"market_attribute")

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.market_attribute_path,
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, is_train=False)

        super(Market1501_Interpretation, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            #print(str(pid))
            if pid == 0:
                p_attribute = -1*torch.ones(size=(26,))
            else:
                p_attribute = self.attribute_dict_all[str(pid)]
                #p_attribute = p_attribute//p_attribute.abs()
                p_attribute = p_attribute.float()
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
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
        if 'age' in self.key_attribute:
            self.key_attribute.remove('age')

        h, w = mat_attribute.shape
        dict_attribute = dict()

        for i in range(h):
            row = mat_attribute.iloc[i:i + 1, :].values.reshape(-1)
            # 1 or 2  ---->   -1 or 1
            dict_attribute[str(int(mat_attribute.index[i]))] = torch.tensor(row[1:].astype(int)) * 2 - 3

        return dict_attribute

    def name_of_attribute(self):
        if self.key_attribute:
            print(self.key_attribute)
            return self.key_attribute
        else:
            assert False