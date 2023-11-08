import os
from pathlib import Path
from torchvision.transforms import Compose
from PIL import Image
from typing import Union
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

class HSTS6(Dataset):
    def __init__(self,
                 size: int,
                 root_dir:Union[Path, str]= "",
                 image_dir:Union[Path, str]="",
                 label_dir:Union[Path, str]="",
                 augment: bool=False):
        super(HSTS6, self).__init__()
        self.size = size
        self.augment = augment
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.images = os.listdir(self.image_path)
        self.labels = os.listdir(self.label_path)
        self.transform = self.build_transform(self.size)
        self.images.sort()
        self.labels.sort()

    def build_transform(self):
        """

        """
        if self.augment:
            Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor()
            ])
        else:
            Compose([])

    def __getitem__(self, item):
        image_item = self.images[item]
        label_item = self.labels[item]
        image_name_path = os.path.join(self.root_dir, self.image_path, image_item)
        label_name_path = os.path.join(self.root_dir, self.label_path, label_item)
        img = Image.open(image_name_path)

        img = self.transform(img)
        with open(label_name_path, "r") as f:
            instance = f.readlines()
        sample = {"image": img, "label": instance}

        return sample

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.labels)


if __name__ == '__main__':
    root_dir = os.path.join("/home/youtian/Documents/pro/pyCode/datasets/HSTS6")
    image_dir = os.path.join("images")
    label_dir = os.path.join("labels")
    hsts = HSTS6(root_dir, image_dir, label_dir, 640, )
    test_set = DataLoader(dataset=hsts, batch_size=4, shuffle=True, num_workers=4, drop_last=False)