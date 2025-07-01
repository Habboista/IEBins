import os
import sys

import torch
from torch import Tensor
import torch.nn as nn

NEWCRF_DIR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(
   NEWCRF_DIR_PATH 
)
from iebins.networks.NewCRFDepth import NewCRFDepth

from custom_classes import BaseDepthModel
from custom_classes import BaseMap
from custom_classes import DepthMap
from custom_classes import DepthMapBatch
from custom_classes import Image
from custom_classes import ImageBatch

DEBUG = False
class IEBins(BaseDepthModel):
    def __init__(self, pretrained=False, max_depth=80.0):
        super().__init__()
        self.name = 'iebins'
        self.max_depth = max_depth
        self.model = nn.DataParallel(
            NewCRFDepth(
                version='large07',
                inv_depth=False,
                max_depth=max_depth,
                pretrained=os.path.join(
                    NEWCRF_DIR_PATH,
                    'swin_large_patch4_window7_224_22k.pth',
                ),
            ),
        )
        if pretrained:
            self.model.load_state_dict(
                torch.load('../IEBins/kittieigen_L.pth', weights_only=False)['model'],
            )

    def forward(
        self,
        image_batch: ImageBatch,
    ) -> Tensor:
        predictions = self.model.module(
            image_batch.to_tensor('cuda'),
        )

        return predictions[0]

    def prepare_for_forward(
        self, 
        image: BaseMap,
    ) -> BaseMap:
        H, W = image.camera['shape']
        if H % 32 > 0:
            pad_H = 32 - H % 32
        else:
            pad_H = 0
        if W % 32 > 0:
            pad_W = 32 - W % 32
        else:
            pad_W = 0
        image = image.transform_by(
            'pad_shape',
            top=pad_H // 2,
            bottom=pad_H - pad_H // 2,
            left=pad_W // 2,
            right=pad_W - pad_W // 2,
        )
        if DEBUG:
            image.save("inference_1_pad.png")
        return image
