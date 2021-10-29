#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File  : add_config.py
@Author: Xiaodong Chen
@Date  : 2020/8/30 21:50
@E-mail  : 1241660907@qq.com or sheldongchen@gmail.com
'''

from fastreid.config import CfgNode as CN


def add_interpretation_config(cfg):
    _C = cfg


    _C.DATALOADER.ATT_RESAMPLE = False





    _C.VISUAL = CN()
    _C.VISUAL.OPEN = False
    _C.VISUAL.GAP_QUERY = 100

    _C.INTERPRETATION = CN()


    _C.INTERPRETATION.FREEZE_LAYERS = ['']  # freeze layers of pretrain model
    _C.INTERPRETATION.PRETRAIN_MODEL = ''

    _C.INTERPRETATION.ATT_PRETRAIN_DICT = ''

    _C.INTERPRETATION.MODEL = CN()
    _C.INTERPRETATION.MODEL.SHARE_LAYER = 3  #   [0,5] , int

    _C.INTERPRETATION.LOSS = CN()
    _C.INTERPRETATION.LOSS.att = 10.0
    _C.INTERPRETATION.LOSS.att_decay = False
    _C.INTERPRETATION.LOSS.interpretation = 1.0
    _C.INTERPRETATION.LOSS.att_lamda = 0.0 # [-inf , 0]ß
    _C.INTERPRETATION.LOSS.threshold = 0.9

    #_C.INTERPRETATION.LOSS.q_att = 1.0  # [-inf , 0]ß



   # Cfg of Interpretation Network  :       g(I)
    _C.INTERPRETATION.I_MODEL = CN()

    _C.INTERPRETATION.I_MODEL.BACKBONE = CN()
    _C.INTERPRETATION.I_MODEL.BACKBONE.ADD_PARAMETER = False
    _C.INTERPRETATION.I_MODEL.BACKBONE.NAME = "build_resnet_backbone"
    _C.INTERPRETATION.I_MODEL.BACKBONE.DEPTH = "50x"
    _C.INTERPRETATION.I_MODEL.BACKBONE.LAST_STRIDE = 1
    # Normalization method for the convolution layers.
    _C.INTERPRETATION.I_MODEL.BACKBONE.NORM = "BN"
    # Mini-batch split of Ghost BN
    _C.INTERPRETATION.I_MODEL.BACKBONE.NORM_SPLIT = 1
    # If use IBN block in backbone
    _C.INTERPRETATION.I_MODEL.BACKBONE.WITH_IBN = False
    # If use SE block in backbone
    _C.INTERPRETATION.I_MODEL.BACKBONE.WITH_SE = False
    # If use Non-local block in backbone
    _C.INTERPRETATION.I_MODEL.BACKBONE.WITH_NL = False
    # If use ImageNet pretrain model
    _C.INTERPRETATION.I_MODEL.BACKBONE.PRETRAIN = True
    # Pretrain model path
    _C.INTERPRETATION.I_MODEL.BACKBONE.PRETRAIN_PATH = ''

    # ---------------------------------------------------------------------------- #
    # REID HEADS options
    # ---------------------------------------------------------------------------- #
    _C.INTERPRETATION.I_MODEL.HEADS = CN()
    _C.INTERPRETATION.I_MODEL.HEADS.NAME = "ADD_AttrHead"

    # Normalization method for the convolution layers.
    _C.INTERPRETATION.I_MODEL.HEADS.NORM = "BN"
    # Mini-batch split of Ghost BN
    _C.INTERPRETATION.I_MODEL.HEADS.NORM_SPLIT = 1
    # Number of identity
    _C.INTERPRETATION.I_MODEL.HEADS.NUM_CLASSES = 23  # _C.INTERPRETATION.NUM_ATT = 23  # num of attribute
    # Input feature dimension
    _C.INTERPRETATION.I_MODEL.HEADS.IN_FEAT = 2048
    # Reduction dimension in head
    _C.INTERPRETATION.I_MODEL.HEADS.REDUCTION_DIM = 512
    # Triplet feature using feature before(after) bnneck
    _C.INTERPRETATION.I_MODEL.HEADS.NECK_FEAT = "before"  # options: before, after
    # Pooling layer type
    _C.INTERPRETATION.I_MODEL.HEADS.POOL_LAYER = "fastavgpool"

    # Classification layer type
    _C.INTERPRETATION.I_MODEL.HEADS.CLS_LAYER = "linear"  # "arcSoftmax" or "circleSoftmax"

    # Margin and Scale for margin-based classification layer
    _C.INTERPRETATION.I_MODEL.HEADS.MARGIN = 0.15
    _C.INTERPRETATION.I_MODEL.HEADS.SCALE = 128
    _C.INTERPRETATION.I_MODEL.HEADS.WITH_BNNECK = False






