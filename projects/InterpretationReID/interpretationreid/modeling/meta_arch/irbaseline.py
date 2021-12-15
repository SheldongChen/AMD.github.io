# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import math
import torch
from torch import nn
import logging
from fastreid.modeling.backbones import build_backbone
from projects.InterpretationReID.interpretationreid.modeling.heads import build_reid_heads
from fastreid.modeling.losses import *
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
import torch.nn.functional as F
from fastreid.modeling.losses.utils import concat_all_gather, euclidean_dist, normalize
from fastreid.utils import comm
from collections import OrderedDict
from fastreid.modeling.backbones.resnest import Bottleneck
from fastreid.utils.weight_init import weights_init_kaiming

@META_ARCH_REGISTRY.register()
class IRBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        self.attribute_sum = torch.zeros((int(cfg.INTERPRETATION.I_MODEL.HEADS.NUM_CLASSES),))
        self.attribute_count = 0

        #loss_iter
        self.interprete_loss_iter = 0

        # backbone
        backbone = build_backbone(cfg)
        #
        i_cfg = cfg.clone()
        i_cfg.defrost()
        i_cfg.MODEL.BACKBONE = i_cfg.INTERPRETATION.I_MODEL.BACKBONE
        i_cfg.MODEL.HEADS = i_cfg.INTERPRETATION.I_MODEL.HEADS

        i_cfg.freeze()

        backbone_plus = build_backbone(i_cfg)
        backbone_att = build_backbone(i_cfg)

        #self.backbone

        if cfg.INTERPRETATION.MODEL.SHARE_LAYER == 0:
            self.backbone_1 = torch.nn.Sequential()
            self.backbone_2 = backbone
            self.backbone_3 = backbone_plus
            self.backbone_att = backbone_att

        else:
            assert 0 <= cfg.INTERPRETATION.MODEL.SHARE_LAYER <= 5

            self.backbone_1 = nn.Sequential()
            self.backbone_2 = nn.Sequential()
            if self._cfg.MODEL.BACKBONE.NAME == "build_resnet_backbone" or self._cfg.MODEL.BACKBONE.NAME == "build_resnest_backbone" :
                name_list = ["conv1","bn1","relu","maxpool"]
                for i in range(1, cfg.INTERPRETATION.MODEL.SHARE_LAYER):
                    name_list.append("layer"+str(i))
            elif self._cfg.MODEL.BACKBONE.NAME == "build_osnet_backbone":
                name_list = ["conv1","maxpool"]
                for i in range(1, cfg.INTERPRETATION.MODEL.SHARE_LAYER):
                    name_list.append("conv"+str(i+1))
            else:
                assert False

            for k,v in backbone.named_children():
                if k in name_list:
                    name = k
                    module = "backbone."+k
                    self.backbone_1.add_module(name, eval(module))
                else:
                    name = k
                    module = "backbone."+k
                    self.backbone_2.add_module(name, eval(module))





            self.backbone_3 = nn.Sequential()
            for k,v in backbone_plus.named_children():
                if k in name_list:
                   pass
                else:
                    name = k
                    module = "backbone_plus."+k
                    self.backbone_3.add_module(name, eval(module))

# backbone_att is designed for demo, pseudo tag and stable re-weight operation, not necessary for explainable ReID
            self.backbone_att = nn.Sequential()
            for k,v in backbone_att.named_children():
                if k in name_list:
                   pass
                else:
                    name = k
                    module = "backbone_att."+k
                    self.backbone_att.add_module(name, eval(module))

            # feature_mask : batch x 23 x 32 x 8
        self.backbone_4 = nn.Sequential()




        self.backbone_4.add_module("conv1",nn.Conv2d(cfg.INTERPRETATION.I_MODEL.HEADS.IN_FEAT, 256, kernel_size=3,padding=1, bias=False))


        self.backbone_4.add_module("conv2",
                                   nn.Conv2d(256, cfg.INTERPRETATION.I_MODEL.HEADS.NUM_CLASSES,
                                             kernel_size=1, bias=False))


        if self._cfg.INTERPRETATION.I_MODEL.BACKBONE.ADD_PARAMETER:
            self.backbone_add = nn.Sequential()
            for i in range(1,10):
                self.backbone_add.add_module("Bottleneck_"+str(i),Bottleneck(inplanes=2048, planes=512, bn_norm='BN', num_splits=1, with_ibn=False, stride=1, downsample=None,radix=1, cardinality=1, bottleneck_width=64, avd=False, avd_first=False, dilation=1, is_first=False,rectified_conv=False, rectify_avg=False, dropblock_prob=0.0, last_gamma=False))




        #logger
        self.logger = logging.getLogger(__name__)

        # head
        self.heads = build_reid_heads(cfg)


        i_cfg.defrost()
        i_cfg.MODEL.HEADS.IN_FEAT *= 1
        i_cfg.freeze()
        self.att_heads = build_reid_heads(i_cfg)
        i_cfg.defrost()
        i_cfg.MODEL.HEADS.IN_FEAT /= 1
        i_cfg.freeze()


        #set for data sample, not necessary for explainable ReID
        if self._cfg.DATASETS.NAMES[0] == "Market1501_And_Interpretation":
            self.attribute_percent = [26.498002663115845, 24.63382157123835, 39.01464713715047, 16.378162450066576,
                                      9.187749667110518, 16.378162450066576, 1.8641810918774968, 7.723035952063914,
                                      42.60985352862849, 11.451398135818907, 2.6631158455392807, 15.046604527296935,
                                      6.125166444740346, 11.451398135818907, 7.456724367509987, 3.9946737683089215,
                                      10.386151797603196, 30.359520639147803]

        elif self._cfg.DATASETS.NAMES[0] == "DukeMTMC_And_Interpretation":
            self.attribute_percent = [64.81481481481481, 16.80911680911681, 42.73504273504273, 31.908831908831907,
                                      4.5584045584045585, 10.968660968660968, 0.14245014245014245, 8.11965811965812,
                                      43.87464387464387, 4.273504273504273, 14.814814814814813, 61.396011396011396,
                                      9.401709401709402, 10.826210826210826, 1.4245014245014245, 1.282051282051282,
                                      6.267806267806268, 7.6923076923076925]

        elif self._cfg.DATASETS.NAMES[0] == "Market1501_Interpretation":
            self.attribute_percent = [26.498002663115845, 24.63382157123835, 85.35286284953395, 60.852197070572565,
                                      39.01464713715047, 16.378162450066576, 9.187749667110518, 16.378162450066576,
                                      1.8641810918774968, 3.861517976031957, 0.2663115845539281, 7.723035952063914,
                                      1.3315579227696404, 42.60985352862849, 32.62316910785619, 11.451398135818907,
                                      2.6631158455392807, 94.8069241011984, 15.046604527296935, 6.125166444740346,
                                      11.451398135818907, 7.456724367509987, 3.9946737683089215, 10.386151797603196,
                                      30.359520639147803, 4.793608521970706]


        elif self._cfg.DATASETS.NAMES[0] == "DukeMTMC_Interpretation":
            self.attribute_percent = [64.81481481481481, 16.80911680911681, 27.20797720797721, 42.73504273504273,
                                      31.908831908831907,
                                      4.5584045584045585, 10.968660968660968, 0.14245014245014245, 1.566951566951567,
                                      8.11965811965812,
                                      43.87464387464387, 4.273504273504273, 14.814814814814813, 14.814814814814813,
                                      15.242165242165242,
                                      61.396011396011396, 9.401709401709402, 1.566951566951567, 10.826210826210826,
                                      1.4245014245014245,
                                      1.282051282051282, 6.267806267806268, 7.6923076923076925]

        else:
            assert False
        self.sample_weight = torch.as_tensor(self.attribute_percent,device='cuda')/100

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)

        real_attributes = self.preprocess_attribute(batched_inputs)

        feature_low = self.backbone_1(images)
        features = self.backbone_2(feature_low) # b x c x h x w , default: 64 x 2048 x 24 x 8
        feature_middle = self.backbone_3(feature_low) # b x 2048 x h x w , default: 64 x 2048 x 24 x 8

        feature_att  = self.backbone_att(feature_low)


        if self._cfg.INTERPRETATION.I_MODEL.BACKBONE.ADD_PARAMETER:

            feature_mask = self.backbone_add(feature_middle)

            feature_mask = self.backbone_4(feature_mask)

        else:
            feature_mask = self.backbone_4(feature_middle)


        b , n , h , w = feature_mask.shape

        feature_mask = ((feature_mask.detach().clone()<0)*(feature_mask)).exp()-1+((feature_mask.detach().clone()>=0)*(feature_mask)+1).pow(0.5)




        #print(self.training)
        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            att_list = list()

            for i in range(0,n):

                dict_heads = self.heads(features * feature_mask[:, i:i+1, :, :], targets,self.training)

                att_heads = self.att_heads(feature_att,targets,self.training)


                #DistributedDataParallel Training , find_unused_parameters=True
                unused_param = None
                for k, v in dict_heads.items():
                    if unused_param == None:
                        unused_param = (torch.zeros_like(v) * v).sum()
                    else:
                        unused_param += (torch.zeros_like(v) * v).sum()
                for k, v in att_heads.items():
                    if unused_param == None:
                        unused_param = (torch.zeros_like(v) * v).sum()
                    else:
                        unused_param += (torch.zeros_like(v) * v).sum()
                dict_heads = dict_heads["features"] + unused_param
                att_list.append(dict_heads)

            outputs = self.heads(features, targets,self.training)



            return {
                "outputs": outputs,
                "targets": targets,
                "att_list":att_list,
                "feature_mask": feature_mask,
                #"fake_attributes":fake_attributes,
                "real_attributes":real_attributes,
                "att_heads":att_heads
            }
        else:
            outputs = self.heads(features,None,self.training)
            att_heads = self.att_heads(feature_att,None,self.training)
            att_list = list()
            for i in range(0,n):
                dict_heads = self.heads(features * feature_mask[:, i:i+1, :, :],None,self.training)
                if isinstance(dict_heads, dict):
                    dict_heads = dict_heads["features"]
                else:
                    dict_heads = dict_heads
                att_list.append(dict_heads)
            #print(feature_mask.shape)
            return {
                "outputs": outputs,
                "att_list":att_list,
                "feature_mask":feature_mask,
                #"fake_attributes":fake_attributes,
                "real_attributes":real_attributes,
                "att_heads":att_heads
            }

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images = (images - self.pixel_mean) / self.pixel_std
        return images

    def preprocess_attribute(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            attributes = batched_inputs["img_attributes"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            attributes = batched_inputs.to(self.device)
        return attributes.type(torch.float32)

    def losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # fmt: off
        outputs           = outs["outputs"]
        gt_labels         = outs["targets"]
        # model predictions
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE

        if "TripletLoss" in loss_names:
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE

        if "CircleLoss" in loss_names:
            loss_dict['loss_circle'] = circle_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self._cfg.MODEL.LOSSES.CIRCLE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE


        loss_dict_attribute = self.attribute_loss(outs,self._cfg)
        loss_dict.update(**loss_dict_attribute)



        loss_dict['loss_attprec'] = 100*self.cross_entropy_sigmoid_loss(outs['att_heads']['cls_outputs'], gt_classes=outs['real_attributes'], sample_weight=self.sample_weight)

        return loss_dict

    def ratio2weight(self,targets, ratio):
        pos_weights = targets * (1 - ratio)
        neg_weights = (1 - targets) * ratio
        weights = torch.exp(neg_weights + pos_weights)

        weights[targets > 1] = 0.0
        #print(weights[0])
        return weights

    def cross_entropy_sigmoid_loss(self, pred_class_logits, gt_classes, sample_weight=None):

        gt_classes = (gt_classes+1)/2


        loss = F.binary_cross_entropy_with_logits(pred_class_logits, gt_classes, reduction='none')

        if sample_weight is not None:
            targets_mask = torch.where(gt_classes.detach() > 0.5,
                                       torch.ones(1, device="cuda"), torch.zeros(1, device="cuda"))  # dtype float32
            weight = self.ratio2weight(targets_mask, sample_weight)
            loss = loss * weight

        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

        loss = loss.sum() / non_zero_cnt
        return loss

    def attribute_loss(self,outs,cfg):
        """
        outs: {
        "outputs": outputs,
        "targets": targets,
        "att_list": att_list,
        "feature_mask":feature_mask,
        "real_attributes": real_attributes }
        """
        self.attribute_sum += outs["real_attributes"].sum(0).cpu()
        self.attribute_count += outs["real_attributes"].size(0)




        norm_feat = cfg.MODEL.LOSSES.TRI.NORM_FEAT

        #fake_attributes = outs["fake_attributes"]
        real_attributes = outs["real_attributes"]
        att_list = outs["att_list"]

        feature_mask = outs["feature_mask"]
        feature_mask_b,feature_mask_n,feature_mask_h,feature_mask_w = feature_mask.size()
        feature_mask = feature_mask.reshape(feature_mask_b,feature_mask_n,feature_mask_h*feature_mask_w)

        # fmt: off
        outputs           = outs["outputs"]

        pred_features     = outputs['features']


        embedding = pred_features



        if norm_feat: embedding = normalize(embedding, axis=-1)

        # For distributed training, gather all features from different process.
        '''
        if comm.get_world_size() > 1:
            all_embedding = concat_all_gather(embedding)
            all_fake_attributes = concat_all_gather(fake_attributes)
            all_real_attributes = concat_all_gather(real_attributes)
            all_att_list = list()
            for att in att_list:
                all_att = concat_all_gather(att)
                all_att_list.append(all_att)


       
        else:
        '''
        all_embedding = embedding

        all_real_attributes = real_attributes # batch x m
        all_att_list = att_list



        batch = all_real_attributes.size(0)

        dist_mat_real = euclidean_dist(all_embedding, all_embedding).detach()  # mat: nxm
        #n,m = dist_mat_real.shape
        #print(n,m)

        dist_mat_fake = torch.zeros_like(dist_mat_real).repeat(cfg.INTERPRETATION.I_MODEL.HEADS.NUM_CLASSES,1,1) # mat: NUM_ATT x n x m

        for i in range(cfg.INTERPRETATION.I_MODEL.HEADS.NUM_CLASSES):
            dist_mat_fake[i] = euclidean_dist(all_att_list[i],all_att_list[i]) # mat: nxm

        dist_mat_fake = dist_mat_fake.permute(1, 2, 0).contiguous()  # mat:   n x m x NUM_ATT

        loss_interpretation_withoutmean = (dist_mat_real-dist_mat_fake.mean(dim=-1)).abs() # n x m
        for i in range(batch):
            loss_interpretation_withoutmean[i,i] = 1e-9 # for numerical stability


        self.interprete_loss_iter += 1

        if self._cfg.INTERPRETATION.LOSS.att_decay:
            div_decay = float(self.interprete_loss_iter)**0.5
        else:
            div_decay = 1.0

        feature_mask_var = feature_mask.var(dim=-1).mean()

        #logger
        gap_real_fake = (loss_interpretation_withoutmean / dist_mat_real).mean() * 100
        if self.interprete_loss_iter % 200 == 0:

            self.logger.info("attribute_sum: {} ,attribute_count: {} \n attribute_sum / count: {}".format(self.attribute_sum,self.attribute_count,(self.attribute_sum/self.attribute_count)/2+0.5))


            #print(loss_interpretation_withoutmean / dist_mat_real)
            self.logger.info("feature_mask_var: \n{} ".format(feature_mask_var))
            self.logger.info("feature mask: \n{} ".format(feature_mask[0,0]))

            self.logger.info("dist_mat_fake 01: \n{} ".format(dist_mat_fake[0,-1]/dist_mat_fake[0,-1].sum()))
            self.logger.info("dist_mat_fake 00: \n{} ".format(dist_mat_fake[0, 0] / dist_mat_fake[0, 0].sum()))
            self.logger.info("gap between real and fake y is {} %".format(gap_real_fake))
            self.logger.info("div_decay is {}".format(div_decay))

        #loss_interpretation_withoutmean = loss_interpretation_withoutmean
        loss_interpretation = loss_interpretation_withoutmean.mean()
        loss_att = self.bool_similarity(dist_mat_fake,all_real_attributes.to(dist_mat_fake.device))



        if gap_real_fake<=self._cfg.INTERPRETATION.LOSS.att_lamda*torch.ones_like(gap_real_fake):
            use_interpretation = 0.0
        else:
            use_interpretation = 1.0



        loss_dict = {
            "loss_att": cfg.INTERPRETATION.LOSS.att*loss_att/div_decay,
            "loss_interpretation": use_interpretation*cfg.INTERPRETATION.LOSS.interpretation * loss_interpretation,
            #"loss_var": 10*nn.ReLU()(loss_var)
        }

        #print(loss_dict)

        return loss_dict

    def bool_similarity(self,dist_mat_fake,all_real_attributes):

        dist_mat_fake_sum = dist_mat_fake.sum(-1).unsqueeze(-1)  # n x m x 1
        dist_mat_fake_percent = dist_mat_fake / dist_mat_fake_sum # n x m x NUM_ATT

        # avoid error
        for i in range(dist_mat_fake.size(0)):
            dist_mat_fake_percent[i,i] = 1.0/dist_mat_fake.size(2)


        assert dist_mat_fake.size() == dist_mat_fake_percent.size()

        all_real_attributes_mat = all_real_attributes.unsqueeze(0) * all_real_attributes.unsqueeze(1)  # n x m x NUM_ATT , belong to {-1, 1} , -1 means different attribute and  1 means same attribute



        all_real_attributes_mat_0_1 = (1.0 - all_real_attributes_mat) / 2.0  # { -1 , 1 } -> { 1 , 0 }


        different_attribute_threshold = (all_real_attributes_mat_0_1.mean(
            dim=-1)).pow(1/2) # clamp for numerical stability ,tensor.size is  n x m , threshold =  pow(1/2)(N / NUM_ATT) , N is the number of different attribute , [0, 1]

        different_attribute_num = all_real_attributes_mat_0_1.sum(
            dim=-1) # n x m , number of different attribute , N

        same_attribute_num = (1.0-all_real_attributes_mat_0_1).sum(
            dim=-1) # n x m , number of same attribute , 23-N

        same_attribute_threshold = (1.0 - different_attribute_threshold) # n x m ,threshold = 1- pow(1/2)( N / NUM_ATT), N is the number of different attribute , [ 0 , 1 ]

        #(dist_mat_fake_percent * all_real_attributes_mat_0_1).sum(-1) >=  different_attribute_threshold

        loss_different_sum = F.relu(different_attribute_threshold - (dist_mat_fake_percent * all_real_attributes_mat_0_1).sum(-1))



        # (dist_mat_fake_percent * (1-all_real_attributes_mat_0_1)).sum(-1) <=  same_attribute_threshold
        loss_same_sum = F.relu((dist_mat_fake_percent * (1.0-all_real_attributes_mat_0_1)).sum(-1) -  same_attribute_threshold)

        assert loss_different_sum.size()==loss_same_sum.size()
        #print(loss_different_sum.size())

        loss_different_sum = loss_different_sum.mean()
        loss_same_sum = loss_same_sum.mean()


        lamda = quadratic_exp(torch.tensor([self._cfg.INTERPRETATION.I_MODEL.HEADS.NUM_CLASSES ],dtype=torch.float32,device=different_attribute_num.device ),different_attribute_num)  # n x m
        lamda_different = (-lamda).exp().unsqueeze(-1)
        lamda_same = (lamda.exp()).unsqueeze(-1)

        loss_different = F.relu( lamda_different *all_real_attributes_mat_0_1 * different_attribute_threshold.unsqueeze(-1) / different_attribute_num.unsqueeze(
            -1).clamp(min=1e-6)  - dist_mat_fake_percent)


        loss_same = F.relu(((1.0 - all_real_attributes_mat_0_1) * dist_mat_fake_percent) -
            lamda_same * same_attribute_threshold.unsqueeze(-1) / same_attribute_num.unsqueeze(-1).clamp(
            min=1e-6))


        ################
        color_mat = all_real_attributes.unsqueeze(0) + all_real_attributes.unsqueeze(
            1)  # n x m x NUM_ATT , belong to {-2, 0, 2} , -2 means all no , 0 means different , 2 means all yes
        color_mat = torch.where(color_mat<=-0.5,torch.ones_like(color_mat), torch.zeros_like(color_mat))



        color_allno = color_mat * dist_mat_fake_percent
        color_notallno = (1.0-color_mat) * dist_mat_fake_percent
        color_allno_threshold =  self._cfg.INTERPRETATION.LOSS.threshold

        #Some fine tuning on Loss, it does not affect the main results. it is not necessary for explainable ReID
        ## loss_color_up and loss_color_down is not necessary !!!! they can be removed !!!
        if self._cfg.DATASETS.NAMES[0]=="Market1501_Interpretation":
            color_mat_notallno_count_up = (1.0 - color_mat[:,:,4:13]).sum(-1).clamp(min=1.0)  # n x m

            loss_color_up = F.relu(color_allno[:,:,4:13]  - color_allno_threshold * (color_notallno[:,:,4:13].sum(-1)/color_mat_notallno_count_up).unsqueeze(-1)).sum(-1).mean()

            color_mat_notallno_count_down = (1.0 - color_mat[:,:,-8:]).sum(-1).clamp(min=1.0)  # n x m
            loss_color_down = F.relu(color_allno[:,:,-8:]  - color_allno_threshold * (color_notallno[:,:,-8:].sum(-1)/color_mat_notallno_count_down).unsqueeze(-1)).sum(-1).mean()

        elif self._cfg.DATASETS.NAMES[0]=="DukeMTMC_Interpretation":

            color_mat_notallno_count_up = (1.0 - color_mat[:,:,3:10]).sum(-1).clamp(min=1.0)  # n x m

            loss_color_up = F.relu(color_allno[:,:,3:10]  - color_allno_threshold * (color_notallno[:,:,3:10].sum(-1)/color_mat_notallno_count_up).unsqueeze(-1)).sum(-1).mean()

            color_mat_notallno_count_down = (1.0 - color_mat[:,:,-8:]).sum(-1).clamp(min=1.0)  # n x m
            loss_color_down = F.relu(color_allno[:,:,-8:]  - color_allno_threshold * (color_notallno[:,:,-8:].sum(-1)/color_mat_notallno_count_down).unsqueeze(-1)).sum(-1).mean()

        elif (self._cfg.DATASETS.NAMES[0] == "DukeMTMC_And_Interpretation" or self._cfg.DATASETS.NAMES[0] == "Market1501_And_Interpretation"):

            color_mat_notallno_count_up = (1.0 - color_mat[:, :, 2:8]).sum(-1).clamp(min=1.0)  # n x m

            loss_color_up = F.relu(color_allno[:, :, 2:8] - color_allno_threshold * (
                        color_notallno[:, :, 2:8].sum(-1) / color_mat_notallno_count_up).unsqueeze(-1)).sum(-1).mean()

            color_mat_notallno_count_down = (1.0 - color_mat[:, :, -7:]).sum(-1).clamp(min=1.0)  # n x m
            loss_color_down = F.relu(color_allno[:, :, -7:] - color_allno_threshold * (
                        color_notallno[:, :, -7:].sum(-1) / color_mat_notallno_count_down).unsqueeze(-1)).sum(-1).mean()

        else:
            assert False

        assert loss_different.size() == loss_same.size()
        #print(loss_different.size())
        loss_different = loss_different.sum(-1).mean()
        loss_same  = loss_same.sum(-1).mean()
        if self.interprete_loss_iter % 200 == 0:

            if self._cfg.DATASETS.NAMES[0] == "Market1501_Interpretation":
                print("*" * 50)
                print(color_allno[0, -1, 4:13])
                print(color_notallno[0, -1, 4:13])
                print(color_allno[0, -1, 4:13],
                      (color_notallno[0, -1, 4:13].sum() / color_mat_notallno_count_up[0, -1]))

                print(color_allno[0, 0, 4:13])
                print(color_notallno[0, 0, 4:13])
                print("*" * 50)

                self.logger.info("all_real_attributes_mat 01:\n{}".format(all_real_attributes_mat[0, -1]))
                self.logger.info("all_real_attributes_mat 00:\n{}".format(all_real_attributes_mat[0, 0]))

                dict_loss_att = {
                        "color_up":{color_mat[0,0,4:13],color_mat[0,-1,4:13]},
                        "color_down": {color_mat[0, 0, -8:], color_mat[0, -1, -8:]},
                        "loss_different_sum": loss_different_sum,
                        "loss_same_sum": loss_same_sum,
                        "loss_same": loss_same,
                        "loss_different": loss_different,
                        "loss_color_up":loss_color_up,
                        "loss_color_down":loss_color_down,
                        #"loss_var":{loss_var,max_loss_var[0],min_loss_var[0]},
                    }
            elif self._cfg.DATASETS.NAMES[0] == "DukeMTMC_Interpretation":
                print("*" * 50)
                print(color_allno[0, -1, 3:10])
                print(color_notallno[0, -1, 3:10])
                print(color_allno[0, -1, 3:10],
                      (color_notallno[0, -1, 3:10].sum() / color_mat_notallno_count_up[0, -1]))

                print(color_allno[0, 0, 3:10])
                print(color_notallno[0, 0, 3:10])
                print("*" * 50)

                self.logger.info("all_real_attributes_mat 01:\n{}".format(all_real_attributes_mat[0, -1]))
                self.logger.info("all_real_attributes_mat 00:\n{}".format(all_real_attributes_mat[0, 0]))
                dict_loss_att = {
                    "color_up": {color_mat[0, 0, 3:10], color_mat[0, -1, 3:10]},
                    "color_down": {color_mat[0, 0, -8:], color_mat[0, -1, -8:]},
                    "loss_different_sum": loss_different_sum,
                    "loss_same_sum": loss_same_sum,
                    "loss_same": loss_same,
                    "loss_different": loss_different,
                    "loss_color_up": loss_color_up,
                    "loss_color_down": loss_color_down,
                    # "loss_var":{loss_var,max_loss_var[0],min_loss_var[0]},
                }
            elif (self._cfg.DATASETS.NAMES[0] == "DukeMTMC_And_Interpretation" or self._cfg.DATASETS.NAMES[
                0] == "Market1501_And_Interpretation"):
                print("*" * 50)
                print(color_allno[0, -1, 2:8])
                print(color_notallno[0, -1, 2:8])
                print(color_allno[0, -1, 2:8],
                      (color_notallno[0, -1, 2:8].sum() / color_mat_notallno_count_up[0, -1]))

                print(color_allno[0, 0, 2:8])
                print(color_notallno[0, 0, 2:8])
                print("*" * 50)

                self.logger.info("all_real_attributes_mat 01:\n{}".format(all_real_attributes_mat[0, -1]))
                self.logger.info("all_real_attributes_mat 00:\n{}".format(all_real_attributes_mat[0, 0]))
                dict_loss_att = {
                    "color_up": {color_mat[0, 0, 2:8], color_mat[0, -1, 2:8]},
                    "color_down": {color_mat[0, 0, -7:], color_mat[0, -1, -7:]},
                    "loss_different_sum": loss_different_sum,
                    "loss_same_sum": loss_same_sum,
                    "loss_same": loss_same,
                    "loss_different": loss_different,
                    "loss_color_up": loss_color_up,
                    "loss_color_down": loss_color_down,
                    # "loss_var":{loss_var,max_loss_var[0],min_loss_var[0]},
                }

            else:
                assert False
            self.logger.info("dict_loss_att:\n{}".format(dict_loss_att))

        # loss_color_up and loss_color_down is not necessary !!!! they can be removed !!!
        return loss_different_sum+10*loss_different+loss_same_sum+10*loss_same+loss_color_up+loss_color_down


def quadratic_exp(A, N):
    # assert N <= A
    A = A.float()
    N = N.float()
    N = N.clamp(min=0.0, max=float(A) - 0.1)
    a = (A - N) * ((N / A).pow(1/2)) + 2.0*2.0*1e-3
    b = N * (1.0 - (N / A).pow(1/2)) + 1.0 *1e-3

    return (a / b).log() / 2.0










