import torch
from torchvision.models import swin_transformer
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool


class SwinBackboneWithFPN(BackboneWithFPN):
    def forward(self, x):
        x = self.body(x)
        for name in x.keys():
            x[name] = x[name].permute([0, 3, 1, 2])
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
    embed_dims = {
        "swin_t":96,
        "swin_s":96,
        "swin_b":128
    }
    if pretrained:
        model = swin_transformer.__dict__[backbone_name](weights="DEFAULT").features
    else:
        model = swin_transformer.__dict__[backbone_name]().features
    # select layers that wont be frozen
    assert 7 >= trainable_layers >= 0
    layers_to_train = [str(7-i) for i in range(8)][:trainable_layers]
    # freeze layers only if pretrained backbone is used

    for name, parameter in model.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = range(4)

    return_layers = {f'{2*(v)+1}': str(v) for v in returned_layers}

    in_channels_stage2 = embed_dims[backbone_name]
    in_channels_list = [in_channels_stage2 * 2 ** i for i in returned_layers]

    return SwinBackboneWithFPN(model, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)


if __name__ == '__main__':
    # test_model = swin_transformer.Swin_T().cuda()
    test_model = swin_fpn_backbone("swin_s",pretrained=True).cuda()
    n_parameters = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    dummy_input = torch.rand(3, 3, 224, 224*2).cuda()
    result = test_model(dummy_input)
    for key, feature_map in result.items():
        print(key, feature_map.shape)

    print(n_parameters)
