from pathlib import Path
import torch.nn as nn

import segmentation_models_pytorch
from segmentation_models_pytorch.encoders._base import EncoderMixin

from segdan.models.davit_v1 import create_encoder
from segdan.utils.utils import Utils

project_root = Path(__file__).resolve().parents[2]
CONFIG_YAML = project_root / "segdan" / "configs" / "medimageinsight_config.yaml"

def register_medimageinsight_to_smp(name="medimageinsight", config_encoder=CONFIG_YAML, pretrained_settings=None):
        """
        Registra el encoder en smp.encoders.encoders[name].
        config_encoder: dict (obligatorio) para pasar a create_encoder() al construir encoder.
        pretrained_settings: dict con keys 'mean','std','input_space','input_range','url' (opcional)
        """

        segmentation_models_pytorch.encoders.encoders[name] = {
            "encoder": MedImageInsightSMPEncoder,
            "params": {
                "config_encoder": config_encoder,
                "pretrained": False,
                "pretrained_weights_path": None
            },
            "pretrained_settings": pretrained_settings or {
                "imagenet": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "input_space": "RGB",
                    "input_range": [0, 1],
                    "url": "" # ?????
                }
            }
        }

class MedImageInsightSMPEncoder(nn.Module, EncoderMixin):
    
    def __init__(self, config_encoder=CONFIG_YAML, **kwargs):
        super().__init__()

        # Read config_encoder
        config = Utils.load_encoder_from_yaml(config_encoder)

        # Load config_encoder
        self.davit = create_encoder(config)

        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels =  list(self.davit.embed_dims)

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth = len(self._out_channels)

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels: int = 3

        print(f"OUT_CHANNELS {self._out_channels}")
        print(f"DEPTH {self._depth}")

    def get_stages(self):

        patch_stride = getattr(self.davit, "PATCH_STRIDE", None)
        if patch_stride is None:
            patch_stride = (4, 2, 2, 2)

        accum = []
        s = 1
        for p in patch_stride:
            s *= p
            accum.append(s)

        stages = {}
    
        stages = {}
        for stride, conv, block in zip(accum, self.davit.convs, self.davit.blocks):
            stages[stride] = [conv, block]

        return stages

    def forward(self, x):
        """Produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
        """
        B = x.size(0)
        print(x.shape)
        size = (x.size(2), x.size(3))
        feat_list = []

        out = x
        for conv, block in zip(self.davit.convs, self.davit.blocks):
            out, size = conv(out, size)   # out: (B, N, C)
            out, size = block(out, size)  # out: (B, N, C)
            H, W = size
            B_, N, C = out.shape

            feat = out.transpose(1, 2).contiguous().view(B_, C, H, W)
            feat_list.append(feat)

        assert len(feat_list) == self._depth, f"expected depth={self._depth}, got {len(feat_list)}"

        return feat_list
    
