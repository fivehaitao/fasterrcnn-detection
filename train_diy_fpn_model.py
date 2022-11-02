from torchvision.models.detection import FasterRCNN
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset.tianchi_dataset import TianchiDataSet
from torch.utils.data.dataloader import DataLoader

from reference_utils import transforms
from reference_utils.engine import train_one_epoch, evaluate


def get_model(pretrain=True, num_classes=21):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrain)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

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

train_dataloader = DataLoader(train_dataset, 2,
                              shuffle=False, num_workers=0,
                              collate_fn=TianchiDataSet.collate_fn)
valid_dataloader = DataLoader(valid_dataset, 2,
                              shuffle=False, num_workers=0,
                              collate_fn=TianchiDataSet.collate_fn)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model(True, len(train_dataset.classes))
model.to(device)

weights = [param for param in model.parameters() if param.requires_grad]

optimizer = torch.optim.SGD(weights, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=3, gamma=0.1)
epoches = 10

for epoch in range(epoches):
    train_one_epoch(model, optimizer, train_dataloader, device, epoch, 50)
    lr_scheduler.step()
    evaluate(model, valid_dataset, device)
