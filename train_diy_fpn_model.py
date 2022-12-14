from model.swin_faster_rcnn import SwinFasterRCNN
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset.tianchi_dataset import TianchiDataSet
from torch.utils.data.dataloader import DataLoader

from reference_utils import transforms
from reference_utils.engine import train_one_epoch, evaluate
from model.backbone.swin_fpn import swin_fpn_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_model(pretrain=True, num_classes=21):
    # backbone_with_fpn = swin_fpn_backbone('swin_t', trainable_layers=4).cuda()
    backbone_with_fpn = resnet_fpn_backbone('resnet50', True, trainable_layers=3)

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],  # 在哪些特征层上进行RoIAlign pooling
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = SwinFasterRCNN(backbone=backbone_with_fpn,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def get_transformer(type="train"):
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    return data_transform[type]


train_dataset = TianchiDataSet(img_root="E:\\tianchidataset\\defect_Images",
                               ann_root="E:\\tianchidataset\\defect_labels_en",
                               data_list_file="E:\\tianchidataset\\set\\traintrain.txt",
                               class_names_file="E:\\tianchidataset\\defect_names.json",
                               transforms=get_transformer("train"))

valid_dataset = TianchiDataSet(img_root="E:\\tianchidataset\\defect_Images",
                               ann_root="E:\\tianchidataset\\defect_labels_en",
                               data_list_file="E:\\tianchidataset\\set\\trainval.txt",
                               class_names_file="E:\\tianchidataset\\defect_names.json",
                               transforms=get_transformer("val"))

train_dataloader = DataLoader(train_dataset, 1,
                              shuffle=False, num_workers=0,
                              collate_fn=TianchiDataSet.collate_fn)
valid_dataloader = DataLoader(valid_dataset, 1,
                              shuffle=False, num_workers=0,
                              collate_fn=TianchiDataSet.collate_fn)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(True, len(train_dataset.classes))
model.to(device)

weights = [param for param in model.parameters() if param.requires_grad]

optimizer = torch.optim.SGD(weights, lr=1e-4,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=3, gamma=0.1)
epoches = 10

for epoch in range(epoches):
    train_one_epoch(model, optimizer, train_dataloader, device, epoch, 50)
    lr_scheduler.step()
    evaluate(model, valid_dataloader, device)

# i = 0
# for img,labels in train_dataloader:
#     print(i)
#     i+=1
#     pass