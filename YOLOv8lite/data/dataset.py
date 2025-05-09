import os
import glob
from typing import Union, List, Tuple
from pathlib import Path
import random
from torch.utils.data.dataset import Dataset
from YOLOv8lite.Utils.process_yaml import yaml_load
from ultralytics.data.augment import v8_transforms, Compose, CopyPaste, RandomPerspective, LetterBox
import numpy as np


class Mosaic:
    """
    mosaic
    """
    def __init__(self, dataset, imgsz=640):
        self.dataset = dataset
        self.imgsz = imgsz
        # 建立单位网格的宽和高
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height

    def get_indexes(self, buffer=True):
        # 从buffer中挑选数据会更快
        if buffer:
            return random.choices(list(self.dataset.buffer), k=3)
        # 从整个数据集上选取
        else:
            return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]

    def _mix_transform(self, labels):
        assert labels.get('rect_shape', None) is None, 'rect and mosaic are mutually exclusive.'
        assert len(labels.get('mix_labels', [])), 'There are no other images for mosaic augment.'
        return self._mosaic4(labels)

    def _mosaic4(self, labels):
        """Create a 2x2 image mosaic."""
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels['mix_labels'][i - 1]
            # Load image
            img = labels_patch['img']
            h, w = labels_patch.pop('resized_shape')

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels['img'] = img4
        return final_labels


    @staticmethod
    def _update_labels(labels, padw, padh):
        """Update labels."""
        nh, nw = labels['img'].shape[:2]
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(nw, nh)
        labels['instances'].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels):
        """Return labels with mosaic border instances clipped."""
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for labels in mosaic_labels:
            cls.append(labels['cls'])
            instances.append(labels['instances'])
        final_labels = {
            'im_file': mosaic_labels[0]['im_file'],
            'ori_shape': mosaic_labels[0]['ori_shape'],
            'resized_shape': (imgsz, imgsz),
            'cls': np.concatenate(cls, 0),
            'instances': Instances.concatenate(instances, axis=0),
            'mosaic_border': self.border}  # final_labels
        final_labels['instances'].clip(imgsz, imgsz)
        good = final_labels['instances'].remove_zero_area_boxes()
        final_labels['cls'] = final_labels['cls'][good]
        return final_labels


    def __call__(self, labels):
        # choice randomly index of three images
        indexes = self.get_indexes()

        # Get images information will be used for Mosaic or MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels['mix_labels'] = mix_labels

        # Mosaic or MixUp
        labels = self._mix_transform(labels)
        labels.pop('mix_labels', None)
        return labels


class YOLOv8Dataset(Dataset):
    """
    Dataset class for loading object detection labels in YOLO format.
    hyp[Path| str]: default.yaml
    """
    def __init__(self,
                 img_path:[Path, str],
                 label_path:[Path, str],
                 imgsz:Union[int, tuple, list],
                 hyp:Union[str, Path],
                 augment:bool=True):

        super(YOLOv8Dataset, self).__init__()
        self.img_path = self.check_path(img_path)
        self.label_path = self.check_path(label_path)

        self.imgae_files = self.get_img_files(self.img_path)

        self.img_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()

        self.imgsz = [imgsz]*2 if isinstance(imgsz, int) else imgsz
        self.hyp = yaml_load(self.check_path(hyp))

        self.augment = augment


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

    def build_transforms(self, hyp=None):
        if self.augment:
            hyp.mosaic = hyp.mosaic if hyp.mosaic else 0
            transforms = v8_transforms(self, hyp)

    @staticmethod
    def img2label_paths(img_paths):
        """Define label paths as a function of image paths."""
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

    def get_labels(self):
        self.label_files = self.img2label_paths(self.img_files)

        return self.label_files

    def check_path(self, path:Union[Path, str]):
        if not isinstance(path, Path):
            path = Path(path)

        return path

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        idx = idx % len(self.img_files)


if __name__ == '__main__':
    a = YOLOv8Dataset("D:\project\yolov8\easy_YOLOv8\YOLOv8lite\coco128\images",
                      "D:\project\yolov8\easy_YOLOv8\YOLOv8lite\coco128\labels",
                      640,
                      "D:\project\yolov8\easy_YOLOv8\YOLOv8lite\configs\\train.yaml")
