from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image


class TianchiDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    def __init__(self, img_root, ann_root, data_list_file, class_names_file, transforms=None):
        # 增加容错能力

        self.img_root = img_root
        self.ann_root = ann_root

        with open(data_list_file) as read:
            self.img_names = [img[:-1] for img in read.readlines()]

        # read class_indict
        json_file = class_names_file
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            self.classes = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.img_root, self.img_names[idx] + '.jpg')
        ann_path = os.path.join(self.ann_root, self.img_names[idx] + '.txt')
        img = Image.open(img_path).convert("RGB")

        with open(ann_path) as ann_file:
            anns = [line[:-1].split(',') for line in ann_file.readlines()]
        boxes = [[int(ann[i]) for i in range(4)] for ann in anns]
        labels = [self.classes.index(ann[4]) for ann in anns]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def get_height_and_width(self, idx):
        # read xml
        img_path = os.path.join(self.img_root, self.img_names[idx] + '.jpg')
        img = Image.open(img_path).convert("RGB")
        return img.size[1], img.size[0]

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


if __name__ == '__main__':
    dataset = TianchiDataSet("E:\\tianchidataset\\defect_Images", "E:\\tianchidataset\\defect_labels_en",
                             "E:\\tianchidataset\\set\\traintrain.txt",
                             "E:\\tianchidataset\\defect_names.json")
    dataset.get_height_and_width(0)
