# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Add additional decode_heads

from .conv_module import ConvModule
from .depthwise_separable_conv_module import DepthwiseSeparableConvModule

__all__ = [
    'ConvModule',
    'DepthwiseSeparableConvModule'
]
