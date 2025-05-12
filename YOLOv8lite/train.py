import torch
import torch.nn as nn
from pathlib import Path
from typing import Union
from YOLOv8lite.network.model import YOLOv8l
from Utils.process_yaml import yaml_load
import time
import glob

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO


class Trainer:
    def __init__(self, cfg_path:Union[Path, str], data_cfg_path:Union[Path, str]):
        # train config file
        self.cfg_dict = yaml_load(cfg_path)
        # dataset config file
        self.data_cfg_path = yaml_load(data_cfg_path)
        self.num_classes = len(self.data_cfg_path.get('names'))

        if self.cfg_dict.get('device') == 'cpu':
            self.device = 'cpu'
            self.cfg_dict['workers'] = 0
        else:
            self.device = "cuda"

        self.model = YOLOv8l(self.num_classes, 16).to(self.device)

        self.validator = None
        self.metrics = None

        # save
        self.name = Path(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
        self.root = Path(self.cfg_dict.get("save_dir")) / self.name
        self.weight_dir = self.root / "weights"

        self.last, self.best = self.weight_dir / "last.pt", self.weight_dir / "best.pt"
        self.save_period = self.cfg_dict.get("save_period")

        self.batch_size = self.cfg_dict.get("batch")
        self.epochs = self.cfg_dict.get("epochs")
        self.start_epoch = 0

        # model and data
        self.data = self.check_dataset(self.cfg_dict.get("data"))
        self.trainset, self.testset = self.get_dataset()

    @staticmethod
    def check_dataset(data_cfg_path:Union[Path, str]):
        # 根据文件名进行拿到configs中的数据配置文件
        data_yaml_path = glob.glob(str(ROOT / "configs" / data_cfg_path), recursive=True)[0]
        if isinstance(data_yaml_path, (Path, str)):
            # 这里将拿到的yaml文件转化成字典
            data_dict = yaml_load(data_yaml_path)
        else:
            raise TypeError("convert yaml to dictionary format ")

        # parse yaml data
        for k in "train", "val":
            if k not in data_dict.keys():
                raise KeyError(f"{k} is not in data dict, check yaml of data please!")
        # add nc parameter for data yaml
        data_dict["nc"] = len(data_dict["names"])





if __name__ == '__main__':
    path = Path("./configs/train.yaml")
    a = yaml_load(path)
    print(type(a.get("data")))


