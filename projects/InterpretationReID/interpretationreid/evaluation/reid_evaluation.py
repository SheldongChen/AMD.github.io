# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import cv2
import copy
import logging
from collections import OrderedDict
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from projects.InterpretationReID.interpretationreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager
from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank import evaluate_rank
from .rerank import re_ranking
from .roc import evaluate_roc
from fastreid.utils import comm
import os
import random
from multiprocessing import Pool,Process
import time
logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        self._num_query = num_query
        self._output_dir = output_dir

        self.features = []
        self.pids = []
        self.camids = []

        # for visualize
        self.imgs = []
        self.imgs_path = []
        self.real_attributes = []
        self.fake_attributes = []
        self.key_attribute = None
        self.att_list = None
        self.feature_mask = []
        self.att_prec = []

        output_dir = self.cfg.OUTPUT_DIR
        if comm.is_main_process() and output_dir:
            PathManager.mkdirs(output_dir)

        rank = comm.get_rank()
        self.logger = setup_logger(output_dir, distributed_rank=rank, name="projects")


    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []

        # for visualize
        self.imgs = []
        self.imgs_path = []
        self.real_attributes = []
        self.fake_attributes = []
        self.key_attribute = None
        self.att_list = None
        self.feature_mask = []
        self.att_prec = []

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.features.append(outputs.cpu())

    def name_of_attribute(self, key_attribute):
        self.key_attribute =  key_attribute

    def process_for_visualize(self, inputs, outs):
        """
        inputs:
        {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
            "img_attributes":p_attribute
        }

        outs:
        {
        "outputs": outputs,  -> tensor : n x 2048
        "att_list": att_list, -> [tensor, tensor, ... tensor ] : n x 2048
        "feature_mask":feature_mask, -> tensor : n x 23 x 7 x 7
        "fake_attributes": fake_attributes,
        "real_attributes": real_attributes

        }


        """


        if self.cfg.VISUAL.OPEN:
            self.imgs.append(inputs["images"].cpu())
        self.imgs_path.extend(inputs["img_paths"])
        self.real_attributes.append(outs["real_attributes"].cpu())
        #self.fake_attributes.append(outs["fake_attributes"].cpu())
        self.feature_mask.append(outs["feature_mask"].cpu())
        if True:
            cls_outputs = outs['att_heads']['cls_outputs'].clone().detach()
            self.att_prec.append(torch.where(cls_outputs>0.5,torch.ones_like(cls_outputs),torch.zeros_like(cls_outputs)).cpu())

        if self.att_list == None:
            self.att_list = outs["att_list"]
            self.len_att_list = len(self.att_list)
        else:
            for i in range(self.len_att_list ):
                self.att_list[i] = torch.cat([self.att_list[i].cpu(),outs["att_list"][i].cpu()],dim=0)

    @staticmethod
    def cal_dist(metric: str, query_feat: torch.tensor, gallery_feat: torch.tensor , out_is_torch=False):
        assert metric in ["cosine", "euclidean"], "must choose from [cosine, euclidean], but got {}".format(metric)
        if metric == "cosine":
            dist = 1 - torch.mm(query_feat, gallery_feat.t())
        else:
            m, n = query_feat.size(0), gallery_feat.size(0)
            xx = torch.pow(query_feat, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(gallery_feat, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist.addmm_(query_feat, gallery_feat.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        if out_is_torch:
            return dist.cpu() #torch.tensor
        else:
            return dist.cpu().numpy()
#TODO CXD Visualizer

    def explain_eval(self,dist_fake_stack, query_attribute, gallery_attribute, lamda=2.0):

        _ , num_att = query_attribute.size()
        num_att = float(num_att)
        n, m, _ = dist_fake_stack.size()

        dist_real_stack = query_attribute.unsqueeze(1) * gallery_attribute.unsqueeze(
            0)  # -1 means different, 1 means same

        assert dist_real_stack.size() == dist_fake_stack.size()
        dist_real_stack_01 = (1.0 - dist_real_stack) / 2.0  # 1 means different, 0 means same
        dist_real_stack_01_reverse = 1.0 - dist_real_stack_01  # 0 means different, 1 means same

        dist_num_different = dist_real_stack_01.sum(-1)  # n x m
        dist_num_same = dist_real_stack_01_reverse.sum(-1)  # n x m

        # dist_fake_stack belong to 0.0~1.0
        dist_fake_stack = dist_fake_stack / dist_fake_stack.sum(-1).unsqueeze(-1)  # n x m x num_att

        # calcuate Precsion of different attribute
        dist_fake_att_different = dist_fake_stack.argsort(dim=-1, descending=True)
        dist_fake_att = (dist_fake_att_different.argsort(dim=-1, descending=False) < dist_num_different.unsqueeze(
            -1)).float()  # n x m x num_att

        Precsion_different_allatt = (dist_fake_att * dist_real_stack_01).sum(0).sum(0)/dist_real_stack_01.sum(0).sum(0)
        self._results["diff_0"] = float(0.0)
        for i in range(Precsion_different_allatt.size(0)):
            self._results["diff_"+str(i+1)+"_"+self.key_attribute[i]] = float(Precsion_different_allatt[i])

        Precsion_different = (dist_fake_att * dist_real_stack_01).sum(-1) / dist_num_different.clamp(min=1.0)
        Precsion_different = Precsion_different.sum() / (n * m - (dist_num_different == 0).float().sum()).clamp(min=1.0)
        self._results["Ex_Precsion_d"] = float(Precsion_different)
        # print(dist_fake_att_different)

        # calcuate Precsion of same attribute
        dist_same_equal_numatt = (dist_num_same.unsqueeze(-1) == num_att).float()  # n x m x 1
        dist_overflow = (dist_fake_stack > lamda / num_att).float() * dist_same_equal_numatt
        dist_fake_att_same = dist_fake_stack.argsort(dim=-1, descending=False)
        dist_fake_att = (dist_fake_att_same.argsort(dim=-1, descending=False) < dist_num_same.unsqueeze(
            -1)).float() - dist_overflow  # n x m x num_att

        Precsion_same_allatt = (dist_fake_att * dist_real_stack_01_reverse).sum(0).sum(0)/dist_real_stack_01_reverse.sum(0).sum(0)
        self._results["same_0"] = float(0.0)
        for i in range(Precsion_different_allatt.size(0)):
            self._results["same_"+str(i+1)+"_"+self.key_attribute[i]] = float(Precsion_same_allatt[i])

        Precsion_same = (dist_fake_att * dist_real_stack_01_reverse).sum(-1) / dist_num_same.clamp(min=1.0)
        Precsion_same = Precsion_same.sum() / (n * m - (dist_num_same == 0).float().sum()).clamp(min=1.0)
        self._results["Ex_Precsion_s"] = float(Precsion_same)

        # print(dist_fake_att_different[0,0])
        raw_cmc = dist_real_stack_01[torch.tensor(range(n)).reshape(-1, 1, 1), torch.tensor(range(m)).reshape(1, -1,
                                                                                                              1), dist_fake_att_different.reshape(
            n, m, int(num_att))]
        tmp_cmc = raw_cmc.cumsum(dim=-1)
        div_cmc = torch.tensor(range(1, int(num_att) + 1)).repeat(n, m, 1)
        tmp_cmc = tmp_cmc / div_cmc * raw_cmc
        AP_d = tmp_cmc.sum(dim=-1) / dist_num_different.clamp(min=1)
        # all_AP_d.append(AP.sum())

        dist_same_equal_numatt = -2 * (dist_same_equal_numatt - 0.5)  # -1 means same attribute == num_att
        dist_fake_att_same = (dist_fake_stack * dist_same_equal_numatt).argsort(dim=-1, descending=False)

        raw_cmc = (dist_real_stack_01_reverse - dist_overflow)[
            torch.tensor(range(n)).reshape(-1, 1, 1), torch.tensor(range(m)).reshape(1, -1,
                                                                                     1), dist_fake_att_same.reshape(n,
                                                                                                                    m,
                                                                                                                    int(
                                                                                                                        num_att))]
        tmp_cmc = raw_cmc.cumsum(dim=-1)
        div_cmc = torch.tensor(range(1, int(num_att) + 1)).repeat(n, m, 1)
        tmp_cmc = tmp_cmc / div_cmc * raw_cmc
        AP_s = tmp_cmc.sum(dim=-1) / dist_num_same.clamp(min=1)

        self._results["Ex_mAP_d"] = AP_d.sum() / (n * m - (dist_num_different == 0).float().sum()).clamp(min=1.0)
        self._results["Ex_mAP_s"] = AP_s.sum() / (n * m - (dist_num_same == 0).float().sum()).clamp(min=1.0)

        return copy.deepcopy(self._results)


    def visualize(self):
        if comm.get_world_size() > 1:
            assert False
            #TODO CXD
        else:
            features = self.features

            # pids = self.pids
            # camids = self.camids

            # for visualize
            if self.cfg.VISUAL.OPEN:
                imgs_path = self.imgs_path
                imgs = self.imgs
            real_attributes = self.real_attributes
            fake_attributes = self.fake_attributes
            key_attribute = self.key_attribute
            att_list = self.att_list
            feature_mask = self.feature_mask



        #att_names: [str,...,str]
        choose_att_names = key_attribute
        if self.cfg.VISUAL.OPEN:
            imgs = torch.cat(imgs, dim=0)
        features = torch.cat(features, dim=0)
        feature_mask = torch.cat(feature_mask,dim=0)
        real_attributes = torch.cat(real_attributes, dim=0)
        #fake_attributes = torch.cat(fake_attributes, dim=0)
        fake_attributes = torch.zeros_like(real_attributes) # adapt
        if self.cfg.VISUAL.OPEN:
            _, _, h_imgs, w_imgs = imgs.size()
        else:
            h_imgs, w_imgs = 384,192



        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        # query_pids = np.asarray(pids[:self._num_query])
        # query_camids = np.asarray(camids[:self._num_query])
        if self.cfg.VISUAL.OPEN:
            query_imgs_path  = imgs_path[:self._num_query]
            query_imgs = imgs[:self._num_query]
        query_real_attributes = real_attributes[:self._num_query]
        query_fake_attributes = fake_attributes[:self._num_query]
        query_feature_mask = feature_mask[:self._num_query]
        query_att_list = []
        for i in range(self.len_att_list):
            query_att_list.append(att_list[i][:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        # gallery_pids = np.asarray(pids[self._num_query:])
        # gallery_camids = np.asarray(camids[self._num_query:])
        if self.cfg.VISUAL.OPEN:
            gallery_imgs_path = imgs_path[self._num_query:]
            gallery_imgs = imgs[self._num_query:]
        gallery_real_attributes = real_attributes[self._num_query:]
        gallery_fake_attributes = fake_attributes[self._num_query:]
        gallery_feature_mask = feature_mask[self._num_query:]
        gallery_att_list = []
        for i in range(self.len_att_list):
            gallery_att_list.append(att_list[i][self._num_query:])


        # Initialize
        if self.cfg.TEST.AQE.ENABLED:
            self.logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)
            for i in range(self.len_att_list):
                query_att_list[i],gallery_att_list[i] = aqe(query_att_list[i] , gallery_att_list[i], qe_time, qe_k, alpha)



        if self.cfg.TEST.METRIC == "cosine":
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)
            for i in range(self.len_att_list):
                query_att_list[i] = F.normalize(query_att_list[i], dim=1)
                gallery_att_list[i] = F.normalize(gallery_att_list[i], dim=1)

        dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features,out_is_torch=True) # n (query) x m
        n,m = dist.size()
        dist_list = [] # [tensor ,..., tensor] -> tensor -> n x m
        for i in range(self.len_att_list):
            dist_list.append(self.cal_dist(self.cfg.TEST.METRIC, query_att_list[i] , gallery_att_list[i],out_is_torch=True))

        #calculate
        dist_list_stack = torch.stack(dist_list,dim=-1) # n x m x NUM_ATT




        #TODO CXD EXP
        fake_gap = (query_fake_attributes.unsqueeze(1)-gallery_fake_attributes.unsqueeze(0)).abs().exp()  # n x m x NUM_ATT
        loss_interpretation_withoutmean = (
                    dist - (dist_list_stack * fake_gap ).mean(dim=-1)).abs()  # n x m
        gap_real_fake = (loss_interpretation_withoutmean / dist).mean() * 100


        self.logger.info("dist:\n {}".format(dist))
        self.logger.info("fake_gap \n {}".format(fake_gap))
        self.logger.info("dist_list_stack \n {}".format(dist_list_stack))


        self.logger.info("gap between real and fake y is {} %".format(gap_real_fake))
        #print("gap between real and fake y is {} %".format(gap_real_fake))

        visiual_average_att(choose_att_names, feature_mask,real_attributes, img_size=(384, 192), output_dir=self.cfg.OUTPUT_DIR,positive=True)
        visiual_average_att(choose_att_names, feature_mask, real_attributes, img_size=(384, 192), output_dir=self.cfg.OUTPUT_DIR, positive=False)
        #print(choose_att_names)
        if self.cfg.VISUAL.OPEN:
            for i_4 in list([2,4,6,8,10,20]): # 2,4,6,8,10,20
            #for i_4 in list([0]):

                self.mkdir_id = str(time.time()).replace(".","_")
                os.mkdir(os.path.join(self.cfg.OUTPUT_DIR, "output_imgs_"+self.mkdir_id))




                for id_query in range(n):
                    if id_query % self.cfg.VISUAL.GAP_QUERY != 0:
                        continue
                # print(query_imgs_path)
                # for id_query in range(2):
                #     id_gallery_choose = gallery_imgs_path.index(list_gallery[id_query])
                #
                #     id_query = query_imgs_path.index(list_query[id_query])

                    _,indices_m = dist[id_query].sort()

                    indices_m = indices_m.numpy().tolist()


                    # type 3
                    list_choosen = [id_gallery_choose, indices_m[0], indices_m[1], indices_m[2]]


                    #path_names:[str,...,str]
                    choose_path_names = [query_imgs_path[id_query].split("/")[-1]]
                    for id_gallery in list_choosen:
                        choose_path_names.append(gallery_imgs_path[id_gallery].split("/")[-1])
                    #imgs: tensor -> n x 3 x 224 x 224
                    choose_imgs = query_imgs[id_query:id_query+1]
                    choose_imgs = torch.cat([choose_imgs,gallery_imgs[list_choosen]],dim=0)

                    #att_mask: tensor -> n x 23 x 224 x 224
                    choose_att_mask = query_feature_mask[id_query:id_query+1]
                    choose_att_mask = torch.cat([choose_att_mask,gallery_feature_mask[list_choosen]],dim=0)

                    #real_attribute: tensor -> 23 x n
                    choose_real_attributes = query_real_attributes[id_query:id_query+1]
                    choose_real_attributes = torch.cat([choose_real_attributes,gallery_real_attributes[list_choosen]],dim=0)
                    choose_real_attributes = choose_real_attributes.t()

                    #fake_attribute: tensor -> 23 x n
                    choose_fake_attributes = query_fake_attributes[id_query:id_query+1]
                    choose_fake_attributes = torch.cat([choose_fake_attributes,gallery_fake_attributes[list_choosen]],dim=0)
                    choose_fake_attributes = choose_fake_attributes.t()

                    #feature_distance: tensor  -> (1 + 23 + 1) x 4
                    choose_feature_distance = dist[id_query:id_query+1,list_choosen] #  1x4

                    for i in range(self.len_att_list):
                        #TODO CXD EXP
                        fake_feature_gap = (choose_fake_attributes[i:i+1,0:1]-choose_fake_attributes[i:i+1,1:]).abs().exp() # 1 x 4
                        fake_feature_distance = dist_list[i][id_query:id_query + 1, list_choosen] * fake_feature_gap  #1 x 4
                        choose_feature_distance = torch.cat([choose_feature_distance,fake_feature_distance],dim=0)



                    fake_sum_feature_distance = choose_feature_distance[1:].mean(dim=0).reshape(1,4)
                    choose_feature_distance = torch.cat( [choose_feature_distance,fake_sum_feature_distance ], dim=0) # 25 x 4



                    visualization_savefig(choose_path_names, choose_att_names, choose_imgs / 255.0, choose_att_mask,
                                          choose_feature_distance, choose_real_attributes, choose_fake_attributes,
                                          output_dir=os.path.join(self.cfg.OUTPUT_DIR, "output_imgs_" + self.mkdir_id),only_hot_map=False)
        #To save memory


        del self.imgs,self.fake_attributes,self.feature_mask,self.att_list
        self.imgs = []
        self.fake_attributes = []
        self.att_list = None
        self.feature_mask = []



        return dist_list_stack,query_real_attributes, gallery_real_attributes # n x m x NUM_ATT

    def evaluate_dist(self,bias_dist,q_att=None,p_att=None,query_real_attributes=None):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            if not comm.is_main_process():
                return {}


        else:
            features = self.features
            pids = self.pids
            camids = self.camids

        features = torch.cat(features, dim=0)
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            self.logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        if self.cfg.TEST.METRIC == "cosine":
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)

        dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)




        if p_att==None and q_att==None:
            raise False
        else:
            #type 2
            bias_dist = bias_dist / bias_dist.sum(-1).unsqueeze(-1)  # n x m x NUM_ATT
            if q_att == 0:
                bias_dist, _ = bias_dist[:,:,2:8].max(-1)  # n x m
            elif q_att == 1:
                bias_dist, _ = bias_dist[:, :, -7:].max(-1)  # n x m
            elif q_att == 2:
                bias_dist_1, _ = bias_dist[:, :, 2:8].max(-1)  # n x m
                bias_dist_2, _ = bias_dist[:, :, -7:].max(-1)  # n x m
                bias_dist = bias_dist_1 + bias_dist_2
            elif q_att == 3:
                bias_dist, _ = bias_dist.max(-1)  # n x m
            elif q_att == 4:

                gallery_fake_attributes = torch.cat(self.att_prec,dim=0)[self._num_query:]
                dist_fake_stack = query_real_attributes.unsqueeze(1) * gallery_fake_attributes.unsqueeze(
                    0)  # -1 means different, 1 means same

                dist_fake_stack = ((-1.0*dist_fake_stack)+1)/2.0  #-1 -> 1 ,1 -> 0,1 means different, 0 means same

                bias_dist = (bias_dist*dist_fake_stack).sum(-1)
            else:
                assert False

            bias_dist = (1.0+p_att*bias_dist)*dist

        if  self.cfg.TEST.RERANK.ENABLED:
            self.logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA
            q_q_dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, query_features)
            g_g_dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, gallery_features)
            re_dist = re_ranking(dist, q_q_dist, g_g_dist, k1, k2, lambda_value)
            re_dist = re_dist * bias_dist.numpy()
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            cmc, all_AP, all_INP = evaluate_rank(re_dist, query_features, gallery_features,
                                                 query_pids, gallery_pids, query_camids,
                                                 gallery_camids, use_distmat=True)
        else:
            #dist = dist * bias_dist.numpy()
            dist = bias_dist.numpy()
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            cmc, all_AP, all_INP = evaluate_rank(dist, query_features, gallery_features,
                                                 query_pids, gallery_pids, query_camids, gallery_camids,
                                                 use_distmat=True)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['New_Rank-{}'.format(r)] = cmc[r - 1]
        self._results['New_mAP'] = mAP
        self._results['New_mINP'] = mINP

        if self.cfg.TEST.ROC_ENABLED:
            scores, labels = evaluate_roc(dist, query_features, gallery_features,
                                          query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["New_TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)


    def evaluate(self):
        if comm.get_world_size() > 1:
            raise False
        else:
            features = self.features
            pids = self.pids
            camids = self.camids

        features = torch.cat(features, dim=0)
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            self.logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        if self.cfg.TEST.METRIC == "cosine":
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)

        dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)

        if self.cfg.TEST.RERANK.ENABLED:
            self.logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA
            q_q_dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, query_features)
            g_g_dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, gallery_features)
            re_dist = re_ranking(dist, q_q_dist, g_g_dist, k1, k2, lambda_value)
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            cmc, all_AP, all_INP = evaluate_rank(re_dist, query_features, gallery_features,
                                                 query_pids, gallery_pids, query_camids,
                                                 gallery_camids, use_distmat=True)
        else:
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            cmc, all_AP, all_INP = evaluate_rank(dist, query_features, gallery_features,
                                                 query_pids, gallery_pids, query_camids, gallery_camids,
                                                 use_distmat=False)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        if self.cfg.TEST.ROC_ENABLED:
            scores, labels = evaluate_roc(dist, query_features, gallery_features,
                                          query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)


def visiual_average_att(att_name, feature_mask, real_attributes,img_size=(384, 192), output_dir="./",positive=True):

    n, c, h, w = feature_mask.size()
    feature_mask = feature_mask.reshape(n, c, h * w)  # n x c x h*w
    if positive:
        feature_mask = feature_mask*((real_attributes+1.0)/2.0).unsqueeze(dim=-1)
    else:
        feature_mask = feature_mask * ((-real_attributes + 1.0) / 2.0).unsqueeze(dim=-1)
    # feature_max, _ = feature_mask.max(dim=-1)  # n x c
    # feature_max = feature_max.unsqueeze(-1)  # n x c  x 1
    # feature_min, _ = feature_mask.min(dim=-1)  # n x c
    # feature_min = feature_min.unsqueeze(-1)  # n x c  x 1
    #
    # feature_mask = (feature_mask - feature_min) / (feature_max - feature_min)  # n x c x h*w

    values, _ = feature_mask.sort(dim=-1, descending=True)
    max_v = values[:, :, h*w//3].unsqueeze(-1)
    feature_mask = torch.where( feature_mask > max_v, torch.ones_like(feature_mask), torch.zeros_like(feature_mask))

    feature_mask = feature_mask.sum(dim=0)  # c x h*w
    feature_max, _ = feature_mask.max(dim=-1)  # c
    feature_max = feature_max.unsqueeze(-1)  # c x 1
    feature_min, _ = feature_mask.min(dim=-1)  # c
    feature_min = feature_min.unsqueeze(-1)  # c x 1

    feature_mask = (feature_mask - feature_min) / (feature_max - feature_min).clamp(min=1e-6)  # c x h*w
    feature_mask = feature_mask.reshape(c, h, w).unsqueeze(0)  # 1 x c x h x w

    feature_mask = torch.nn.Upsample(size=img_size, scale_factor=None, mode='bilinear', align_corners=False)(
        feature_mask)
    feature_mask = feature_mask[0]
    len_att = len(att_name)
    num_col = 5 #int(len_att)

    num_raw = ((len_att - 1) // num_col + 1) * 2
    plt.figure(figsize=(num_col, num_raw), dpi=224)

    for i in range(0, num_raw, 2):
        for j in range(num_col):
            if i // 2 * num_col + j >= len_att:
                break
            plt.subplot(num_raw, num_col, i * num_col + j + 1)
            plt.axis('off')  # 去掉坐标轴
            plt.text(0.5, 0.5, att_name[i // 2 * num_col + j], fontsize=8, ha='center', va="center")

    for i in range(1, num_raw, 2):
        for j in range(num_col):
            if i // 2 * num_col + j >= len_att:
                break
            plt.subplot(num_raw, num_col, i * num_col + j + 1)
            plt.axis('off')  # 去掉坐标轴

            plt.imshow(feature_mask[i // 2 * num_col + j], cmap='nipy_spectral')
            #plt.imshow(cv2.applyColorMap(np.uint8(255 * feature_mask[i // 2 * num_col + j]), cv2.COLORMAP_JET))
            if j==num_col-1:
                plt.colorbar()

    plt.tight_layout()
    if positive:
        plt.savefig(os.path.join(output_dir, str(time.time())+"_positive_att_mask.jpg"))
    else:
        plt.savefig(os.path.join(output_dir, str(time.time())+"_negative_att_mask.jpg"))

    plt.clf()
    plt.close()


def visualization_savefig(path_names, att_names, imgs, att_mask, feature_distance, real_attribute, fake_attribute,output_dir="./",only_hot_map=False):
    """
    path_names:[str,...,str]
    att_names: [str,...,str]
    imgs: tensor -> n x 3 x 224 x 224
    att_mask: tensor -> n x 23 x 224 x 224
    feature_distance: tensor  -> (1+23+1) x 4
    real_attribute: tensor -> 23 x n
    fake_attribute: tensor -> 23 x n
    output_dir: default: "./"
    """
    #self.cfg.MODEL.HEADS.IN_FEAT
    # b_att,c_att,h_att,w_att = att_mask.size()
    # att_mask_use = torch.zeros(size=(b_att,c_att//IN_FEAT,h_att,w_att))
    # for i in range(c_att//IN_FEAT):
    #     att_mask_use[:,i,:,:] = att_mask[:,i*IN_FEAT:(i+1)*IN_FEAT,:,:].mean(dim=1)
    # att_mask = att_mask_use
    torch.save({'path_names':path_names,'imgs':imgs,'att_mask':att_mask,'real_attribute':real_attribute}, os.path.join(output_dir, path_names[0].replace(".", "_") + "mask.pth"))

    _,_,h, w = imgs.size()
    att_mask = torch.nn.Upsample(size=(h, w), scale_factor=None, mode='bilinear', align_corners=False)(att_mask)
    # rank of feature
    feature_distance_for_rank = feature_distance[1:-1]
    _, indices = feature_distance_for_rank.sort(dim=0, descending=True)
    _, indices = indices.sort(dim=0)
    indices = indices + 1


    num_low = len(att_names) + 2  # 25
    num_col = len(path_names) * 2  # 10
    plt.figure(figsize=(num_col, num_low), dpi=224)

    # name of pic
    for i, j in zip([0] * (num_col // 2), list(range(0, num_col, 2))):
        plt.subplot(num_low, num_col, i * num_col + j + 2)
        plt.axis('off')  # 去掉坐标轴
        plt.text(0.5, 0.5, path_names[j // 2], fontsize=6, ha='center', va="center")

    # name of title
    plt.subplot(num_low, num_col, num_col + 1)
    plt.axis('off')  # 去掉坐标轴
    str_show = "orginal"
    plt.text(0.5, 0.5, str_show, fontsize=14, ha='center', va="center")

    # name of att
    for i, j in zip(list(range(2, num_low, 1)), [0] * (num_low - 2)):
        plt.subplot(num_low, num_col, i * num_col + j + 1)
        plt.axis('off')  # 去掉坐标轴
        str_show = att_names[i - 2]
        str_show += "\n real_att: " + str(int(real_attribute[i - 2, 0]))
        str_show += "\n fake_att: " + str(float(fake_attribute[i - 2, 0]))[:6]
        plt.text(0.5, 0.5, str_show, fontsize=6, ha='center', va="center")

    # image
    for i in list(range(1, num_low, 1)):
        for j in list(range(0, num_col, 2)):
            plt.subplot(num_low, num_col, i * num_col + j + 2)
            plt.axis('off')  # 去掉坐标轴
            if i == 1:
                plt.imshow(imgs[j // 2].permute(1, 2, 0))
            else:
                min_att_mask = att_mask[j // 2, i - 2].min()
                max_att_mask = att_mask[j // 2, i - 2].max()
                att_mask_use = (att_mask[j // 2, i - 2] - min_att_mask) / (max_att_mask - min_att_mask)  # 0~1


                if only_hot_map:
                    # For a better viewing experience
                    plt.imshow(att_mask_use.pow(2), cmap='nipy_spectral')
                    #plt.imshow(att_mask_use, cmap='nipy_spectral')
                else:

                    plt.imshow((imgs[(j // 2)]).permute(1, 2, 0))
                    plt.imshow(att_mask_use.pow(2), cmap='nipy_spectral',alpha=0.4)
                    #plt.imshow(att_mask_use, cmap='nipy_spectral',alpha=0.4)


    # distance
    for j in list(range(2, num_col, 2)):
        # all_gap = 0
        for i in list(range(1, num_low, 1)):

            plt.subplot(num_low, num_col, i * num_col + j + 1)
            plt.axis('off')  # 去掉坐标轴
            # print(i-1,j//2-1)
            if i == 1:
                #TODO CXD ABS .abs()
                gap_fake_real = (feature_distance[-1, j // 2 - 1] - feature_distance[i - 1, j // 2 - 1]) / \
                                feature_distance[i - 1, j // 2 - 1] * 100
                str_show = str(float(feature_distance[i - 1, j // 2 - 1]))[:8] + "\n\n gap: " + str(
                    float(gap_fake_real))[:6] + "%"
                str_show += "\n var:"+str(float((feature_distance[1:-1, j // 2 - 1]/feature_distance[-1:, j // 2 - 1]).var()))[:8]

                str_show += "\n real_num:"+str(float(((1-(real_attribute[:,j // 2] * real_attribute[:,0]))/2).sum()))[:6]
                plt.text(0.5, 0.5, str_show, fontsize=6, ha='center', va="center")
            else:

                outstand_gap = feature_distance[i - 1, j // 2 - 1] / feature_distance[-1, j // 2 - 1] * 100
                rank_att = str(float(indices[i - 2, j // 2 - 1]))
                # all_gap+=outstand_gap
                str_show = str(float(feature_distance[i - 1, j // 2 - 1]))[:8] + "\n" + str(float(outstand_gap))[
                                                                                        :6] + "%\n rank: " + rank_att
                str_show += "\n real_att: " + str(int(real_attribute[i - 2, j // 2]))
                #str_show += "\n fake_att: " + str(float(fake_attribute[i - 2, j // 2]))[:6]
                plt.text(0.5, 0.5, str_show, fontsize=6, ha='center', va="center")

        # print(all_gap)
    if only_hot_map:
        plt.savefig(os.path.join(output_dir, path_names[0].replace(".", "_") + "only_hot_map.jpg"))
    else:
        plt.savefig(os.path.join(output_dir, path_names[0].replace(".", "_") + "notonly_hot_map.jpg"))
    torch.save(feature_distance,os.path.join(output_dir, path_names[0].replace(".", "_") + ".pth"))
    plt.clf()
    plt.close()
