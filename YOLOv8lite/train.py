import torch
import torch.nn as nn
from pathlib import Path
from typing import Union
from YOLOv8lite.network.model import YOLOv8l
from Utils.process_yaml import yaml_load
import time


class Trainer(object):
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
        self.trainset, self.testset = self.get_dataset()


    def train(self):
        ...

def train(cfg):
    model = ...
    data = cfg.data
    trainer = Trainer()
    trainer.train()

if __name__ == '__main__':

    path = Path("./configs/dataset.yaml")
    print(yaml_load(path))


