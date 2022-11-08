import torch
from model.backbone import swin_transformer
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from collections import OrderedDict




class SwinBackboneWithFPN(BackboneWithFPN):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extra_blocks=None):
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = backbone
        self.return_layers = return_layers
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x, self.return_layers)
        x = self.fpn(x)
        return x


def swin_fpn_backbone(
        backbone_name,
        pretrained=False,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,
        trainable_layers=3,
        returned_layers=None,
        extra_blocks=None,
        out_channels=256
):

    model = swin_transformer.__dict__[backbone_name]()
    # select layers that wont be frozen
    assert 4 >= trainable_layers >= 0
    layers_to_train = ['stage4', 'stage3', 'stage2', 'stage1'][:trainable_layers]
    # freeze layers only if pretrained backbone is used

    for name, parameter in model.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]

    return_layers = {f'stage{k}': str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = model.hidden_dim
    in_channels_list = [in_channels_stage2 * 2 ** (i-1) for i in returned_layers]

    return SwinBackboneWithFPN(model, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


if __name__ == '__main__':
    # test_model = swin_transformer.Swin_T().cuda()
    test_model = swin_fpn_backbone("swin_t").cuda()
    n_parameters = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    dummy_input = torch.rand(3, 3, 224, 224*2).cuda()
    result = test_model(dummy_input)
    for key, feature_map in result.items():
        print(key, feature_map.shape)
    # flops, params = profile(test_model, inputs=(dummy_input, ))
    # print(params)
    # print(flops)
    print(n_parameters)
