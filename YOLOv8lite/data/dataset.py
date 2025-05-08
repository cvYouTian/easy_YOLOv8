import os
import glob
from typing import Union
from pathlib import Path
from torch.utils.data.dataset import Dataset
from YOLOv8lite.Utils.process_yaml import yaml_load


class YOLOv8Dataset(Dataset):
    """
    Dataset class for loading object detection labels in YOLO format.
    hyp[Path| str]: default.yaml
    """

    def __init__(self,
                 img_path:[Path, str],
                 label_path:[Path, str],
                 imgsz:Union[int, tuple, list],
                 hyp:Union[str, Path]):

        super(YOLOv8Dataset, self).__init__()
        self.img_path = self.check_path(img_path)
        self.label_path = self.check_path(label_path)

        self.imgae_files = self.get_img_files(self.img_path)

        self.img_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()

        self.imgsz = [imgsz]*2 if isinstance(imgsz, int) else imgsz
        self.hyp = yaml_load(self.check_path(hyp))

    # from ultraliytics
    def get_img_files(self, img_path):
        """Read image files."""
        f = []  # image files
        for p in img_path if isinstance(img_path, list) else [img_path]:
            p = Path(p)  # os-agnostic
            if p.is_dir():  # dir
                f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                # F = list(p.rglob('*.*'))  # pathlib
            elif p.is_file():  # file
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
        im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ("jpg", "png", "jpeg"))

        return im_files

    @staticmethod
    def img2label_paths(img_paths):
        """Define label paths as a function of image paths."""
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

    def get_labels(self):
        self.label_files = self.img2label_paths(self.img_files)
        print(self.label_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        idx = idx % len(self.img_files)

    def check_path(self, path:Union[Path, str]):
        if not isinstance(path, Path):
            path = Path(path)

        return path




if __name__ == '__main__':
    a = YOLOv8Dataset("D:\project\yolov8\easy_YOLOv8\YOLOv8lite\coco128\images",
                      "D:\project\yolov8\easy_YOLOv8\YOLOv8lite\coco128\labels",
                      640,
                      "D:\project\yolov8\easy_YOLOv8\YOLOv8lite\configs\\train.yaml")

    a.get_labels()