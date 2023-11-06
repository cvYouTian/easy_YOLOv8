import os
from PIL import Image
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

class HSTS6(Dataset):
    def __init__(self, root_dir, image_dir, label_dir, transform):
        super(HSTS6, self).__init__()
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.images = os.listdir(self.image_path)
        self.labels = os.listdir(self.label_path)
        self.transform = transform
        self.images.sort()
        self.labels.sort()

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
    root_dir = ""
    image_dir = ""
    label_dir = ""
    transforms = torchvision.transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    mydata = HSTS6()
    test_set = DataLoader(dataset=HSTS6, batch_size=4, shuffle=True, num_workers=4, drop_last=False)