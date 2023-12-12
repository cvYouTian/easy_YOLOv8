import os
from pathlib import Path
from PIL import Image
from typing import Union
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class HSTS6(Dataset):
    def __init__(self,
                 size: int,
                 root_dir: Union[Path, str] = "",
                 image_dir: Union[Path, str] = "",
                 label_dir: Union[Path, str] = "",
                 augment: bool = False):
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
        self.transform = self.build_transform()
        self.images.sort()
        self.labels.sort()

    def build_transform(self):
        """define your method of augment.
        like:
            if self.augment:
                return trianing transform
            else:
                return val transform
        """
        if self.augment:
            return transforms.Compose([
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor()
            ])
        # raise NotImplementedError

    def image_files(self, img_path: str):
        """Read image files

        Two case:
            1.Folder to save images
            2.File with image path

        """
        try:
            # 提前定义一个存储图片绝对路径的列表
            l: list = list()
            for p in img_path if isinstance(img_path, list) else list(img_path):
                p = Path(p)
                if p.is_dir():
                    # 将rglob产生的迭代器转化成列表，内容时图片的路径
                    l += list(p.rglob("*.*"))
                elif p.is_file():
                    with open(p) as f:
                        f = f.read().strip().splitlines()
                        # to transform a abs patn from reference path .
                        l += list(p.parent / x.lstrip(os.sep) for x in f)
                else:
                    raise FileNotFoundError("%s does not exist！！" % img_path)

        except Exception as e:
            raise e

    def __getitem__(self, item):
        image_item = self.images[item]
        print(image_item)
        label_item = self.labels[item]
        image_name_path = os.path.join(self.root_dir, self.image_path, image_item)
        print(image_name_path)
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
    input_size = 640
    root_dir = Path("/home/youtian/Documents/pro/pyCode/datasets/HSTS6")
    image_dir = Path("images")
    label_dir = Path("labels")
    hsts = HSTS6(input_size, root_dir, image_dir, label_dir)
    a = hsts[1]
    print(a)

    # test_set = DataLoader(dataset=hsts, batch_size=4, shuffle=True, num_workers=4, drop_last=False)
