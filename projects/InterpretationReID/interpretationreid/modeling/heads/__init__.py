# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import REID_HEADS_REGISTRY, build_reid_heads

# import all the meta_arch, so they will be registered
from .add_linear_head import ADD_LinearHead
from .add_bnneck_head import ADD_BNneckHead
from .add_reduction_head import ADD_ReductionHead
from .add_fastatt_head import ADD_AttrHead