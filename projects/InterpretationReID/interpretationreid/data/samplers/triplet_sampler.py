# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import copy
import itertools
from collections import defaultdict
from typing import Optional

import numpy as np
from torch.utils.data.sampler import Sampler

from fastreid.utils import comm
import torch

def no_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class BalancedIdentitySampler(Sampler):
    def __init__(self, data_source: str, batch_size: int, num_instances: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = sorted(list(self.pid_index.keys()))
        self.num_identities = len(self.pids)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            # Shuffle identity list
            identities = np.random.permutation(self.num_identities)

            # If remaining identities cannot be enough for a batch,
            # just drop the remaining parts
            drop_indices = self.num_identities % self.num_pids_per_batch
            if drop_indices: identities = identities[:-drop_indices]

            ret = []
            for kid in identities:
                i = np.random.choice(self.pid_index[self.pids[kid]])
                _, i_pid, i_cam = self.data_source[i]
                ret.append(i)
                pid_i = self.index_pid[i]
                cams = self.pid_cam[pid_i]
                index = self.pid_index[pid_i]
                select_cams = no_index(cams, i_cam)

                if select_cams:
                    if len(select_cams) >= self.num_instances:
                        cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                    else:
                        cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)
                    for kk in cam_indexes:
                        ret.append(index[kk])
                else:
                    select_indexes = no_index(index, i)
                    if not select_indexes:
                        # Only one image for this identity
                        ind_indexes = [0] * (self.num_instances - 1)
                    elif len(select_indexes) >= self.num_instances:
                        ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                    else:
                        ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                    for kk in ind_indexes:
                        ret.append(index[kk])

                if len(ret) == self.batch_size:
                    yield from ret
                    ret = []


class NaiveIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source: str, batch_size: int, num_instances: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = sorted(list(self.pid_index.keys()))
        self.num_identities = len(self.pids)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            avai_pids = copy.deepcopy(self.pids)
            batch_idxs_dict = {}

            batch_indices = []
            while len(avai_pids) >= self.num_pids_per_batch:
                selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False).tolist()
                for pid in selected_pids:
                    # Register pid in batch_idxs_dict if not
                    if pid not in batch_idxs_dict:
                        idxs = copy.deepcopy(self.pid_index[pid])
                        if len(idxs) < self.num_instances:
                            idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                        np.random.shuffle(idxs)
                        batch_idxs_dict[pid] = idxs

                    avai_idxs = batch_idxs_dict[pid]
                    for _ in range(self.num_instances):
                        batch_indices.append(avai_idxs.pop(0))

                    if len(avai_idxs) < self.num_instances: avai_pids.remove(pid)

                assert len(batch_indices) == self.batch_size, f"batch indices have wrong " \
                                                              f"length with {len(batch_indices)}!"
                yield from batch_indices
                batch_indices = []

class NaiveIdentitySamplerForCrossDomain(Sampler):
    """
    sample N identities with attribute, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid, p_attribute).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source: str, batch_size: int, num_instances: int, seed: Optional[int] = None ,cfg = None):

        self._cfg = cfg.clone()
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        #TODO CXD
        #1_backpack   | 2_bag   | 3_downblack   | 4_downblue   | 5_downbrown   | 6_downgray   | 7_downgreen   | 8_downwhite   | 9_gender   | 10_handbag   | 11_hat   | 12_upblack   | 13_upblue   | 14_upgray   | 15_upgreen   | 16_uppurple   | 17_upred   | 18_upwhite
        # [2:8] downcolor , [-7:] upcolor
        if self._cfg.DATASETS.NAMES[0] == "Market1501_And_Interpretation":
            self.attribute_percent = [26.498002663115845, 24.63382157123835, 39.01464713715047, 16.378162450066576, 9.187749667110518, 16.378162450066576, 1.8641810918774968, 7.723035952063914, 42.60985352862849, 11.451398135818907, 2.6631158455392807, 15.046604527296935, 6.125166444740346, 11.451398135818907, 7.456724367509987, 3.9946737683089215, 10.386151797603196, 30.359520639147803]
            self.pow_y = 1.5
        elif self._cfg.DATASETS.NAMES[0] == "DukeMTMC_And_Interpretation":
            self.attribute_percent = [64.81481481481481, 16.80911680911681, 42.73504273504273, 31.908831908831907, 4.5584045584045585, 10.968660968660968, 0.14245014245014245, 8.11965811965812, 43.87464387464387, 4.273504273504273, 14.814814814814813, 61.396011396011396, 9.401709401709402, 10.826210826210826, 1.4245014245014245, 1.282051282051282, 6.267806267806268, 7.6923076923076925]
            self.pow_y = 1.2
        elif self._cfg.DATASETS.NAMES[0] == "Market1501_Interpretation":
            self.attribute_percent = [26.498002663115845, 24.63382157123835, 85.35286284953395, 60.852197070572565, 39.01464713715047, 16.378162450066576, 9.187749667110518, 16.378162450066576, 1.8641810918774968, 3.861517976031957, 0.2663115845539281, 7.723035952063914, 1.3315579227696404, 42.60985352862849, 32.62316910785619, 11.451398135818907, 2.6631158455392807, 94.8069241011984, 15.046604527296935, 6.125166444740346, 11.451398135818907, 7.456724367509987, 3.9946737683089215, 10.386151797603196, 30.359520639147803, 4.793608521970706]
            self.pow_y = 1.2

        elif self._cfg.DATASETS.NAMES[0] == "DukeMTMC_Interpretation":
            self.attribute_percent = [64.81481481481481, 16.80911680911681, 27.20797720797721, 42.73504273504273, 31.908831908831907,
             4.5584045584045585, 10.968660968660968, 0.14245014245014245, 1.566951566951567, 8.11965811965812,
             43.87464387464387, 4.273504273504273, 14.814814814814813, 14.814814814814813, 15.242165242165242,
             61.396011396011396, 9.401709401709402, 1.566951566951567, 10.826210826210826, 1.4245014245014245,
             1.282051282051282, 6.267806267806268, 7.6923076923076925]
            self.pow_y = 1.2
        else:
            assert False

        self.reverse = torch.tensor([ -1.0 if i>50.0 else 1.0   for i in self.attribute_percent])
        self.attribute_percent_adjust = [100.0-i if i>50.0 else i   for i in self.attribute_percent]

        print("self.attribute_percent_adjust = {}".format(self.attribute_percent_adjust))
        print("self.reverse                  = {}".format(self.reverse))

        self.len_attribute = len(self.attribute_percent)
        #self.attribute_percent = torch.tensor(self.attribute_percent)
        pid_sampling_ratios_min = 1000
        pid_sampling_ratios_max = 0
        self.pid_sampling_ratios =  defaultdict(int)

        self.pid_attribute = defaultdict(int)

        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)
            if isinstance(self.pid_attribute[pid],int) :
                self.pid_attribute[pid] = info[3]


                sampling_ratios = [ 1.0/self.attribute_percent_adjust[i]  for i in range(self.len_attribute) if (info[3]*self.reverse)[i]>0 ]
                len_sampling_ratios = float(len(sampling_ratios ))

                sampling_ratios = len_sampling_ratios / torch.tensor(sampling_ratios).sum().float()

                self.pid_sampling_ratios[pid] = 100.0/sampling_ratios
                if self.pid_sampling_ratios[pid]<=pid_sampling_ratios_min:
                    pid_sampling_ratios_min = self.pid_sampling_ratios[pid]
                if self.pid_sampling_ratios[pid] >= pid_sampling_ratios_max:
                    pid_sampling_ratios_max = self.pid_sampling_ratios[pid]

        for key,value in  self.pid_sampling_ratios.items():
            #print("pid = {}, value/pid_sampling_ratios_min  = {}".format(key,value/pid_sampling_ratios_min ))
            self.pid_sampling_ratios[key] = int(pow(int(value/pid_sampling_ratios_min),self.pow_y))

        print("pid_sampling_ratios_min  = {}".format(pid_sampling_ratios_min))
        print("pid_sampling_ratios_max  = {}".format(pid_sampling_ratios_max))
        self.pids= sorted(list(self.pid_index.keys()))

        self.pids_weight = []
        self.pids_weight_order = defaultdict(int)
        order_i = 0
        for pid in self.pids:
            self.pids_weight.append(self.pid_sampling_ratios[pid])
            self.pids_weight_order[pid] =  order_i
            order_i += 1
        print(self.pids_weight)
        print(self.pids_weight_order)

        #
        # self.pids = list()
        # for pid in self.pids_withoutratio:
        #     self.pids.extend([pid]*self.pid_sampling_ratios[pid])

        self.num_identities = len(self.pids)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            avai_pids = copy.deepcopy(self.pids)
            avai_pids_for_count = copy.deepcopy(self.pids)
            pid_weight = np.array(copy.deepcopy(self.pids_weight))
            #pid_weight_dict = copy.deepcopy(self.pid_sampling_ratios)

            batch_idxs_dict = {}
            batch_indices = []
            while len(avai_pids_for_count) >= self.num_pids_per_batch:
                selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False,p=pid_weight/pid_weight.sum()).tolist()
                for pid in selected_pids:
                    # Register pid in batch_idxs_dict if not
                    if pid not in batch_idxs_dict:
                        idxs = copy.deepcopy(self.pid_index[pid])
                        if len(idxs) < self.num_instances:
                            idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                        np.random.shuffle(idxs)
                        batch_idxs_dict[pid] = idxs

                    avai_idxs = batch_idxs_dict[pid]
                    for _ in range(self.num_instances):
                        batch_indices.append(avai_idxs.pop(0))

                    if len(avai_idxs) < self.num_instances:
                        pid_weight[self.pids_weight_order[pid]] -= 1
                        assert pid_weight[self.pids_weight_order[pid]]>=0
                        # pid_weight_dict[pid] -= 1
                        # assert pid_weight_dict[pid] >= 0
                        if pid_weight[self.pids_weight_order[pid]]==0:
                            avai_pids_for_count.remove(pid)
                        del batch_idxs_dict[pid]

                assert len(batch_indices) == self.batch_size, f"batch indices have wrong " \
                                                              f"length with {len(batch_indices)}!"
                yield from batch_indices
                batch_indices = []
