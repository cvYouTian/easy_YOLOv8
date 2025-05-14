import torch
from pathlib import Path
from types import SimpleNamespace
from YOLOv8lite.data.dataset import YOLODataset
from typing import Union
from YOLOv8lite.network.model import YOLOv8l
from Utils.process_yaml import yaml_load
import time
import glob

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils.loss import v8DetectionLoss


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO

class DetectionModel:
    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None):
        self.yaml = cfg if isinstance(cfg, dict) else yaml_load(cfg)

    def _perdict_augment(self):
        ...

    def init_criterion(self):
        return v8DetectionLoss

class Trainer:
    def __init__(self, cfg_path:Union[Path, str], data_cfg_path:Union[Path, str]):
        # train config file
        self.train_cfg_dict = yaml_load(cfg_path)

        self.epoch_time_start = time.time()
        self.train_time_start = time.time()

        # dataset config file
        self.data_cfg_path = yaml_load(data_cfg_path)
        self.num_classes = len(self.data_cfg_path.get('names'))

        if self.train_cfg_dict.get('device') == 'cpu':
            self.device = 'cpu'
            self.train_cfg_dict['workers'] = 0
            self.world_size = 0
        else:
            self.device = "cuda"
            self.world_size = 1

        self.validator = None
        self.metrics = None

        # save
        self.name = Path(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
        self.root = Path(self.train_cfg_dict.get("save_dir")) / self.name
        self.weight_dir = self.root / "weights"

        self.last, self.best = self.weight_dir / "last.pt", self.weight_dir / "best.pt"
        self.save_period = self.train_cfg_dict.get("save_period")

        self.batch_size = self.train_cfg_dict.get("batch")
        self.epochs = self.train_cfg_dict.get("epochs")
        self.start_epoch = 0

        # model and data
        self.model = YOLOv8l(self.num_classes, 16).to(self.device)

        # convert path to absolute path
        self.data = self.check_dataset(self.train_cfg_dict.get("data"))
        self.trainset, self.testset = self.data.get("train"), self.data.get("test")

        # training...
        self.do_trian()

    def get_dataloader(self, dataset_path, batch_size, mode):
        assert mode in ['train', 'val']

        dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffe = mode == "train"

        workers = self.train_cfg_dict.workers if mode == 'train' else self.train_cfg_dict.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffe,)


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

        # add the number of the classes parameter for data yaml
        data_dict["nc"] = len(data_dict["names"])

        # override the path of dataset with the abosulte path
        path = Path(data_dict.get("path"))
        if not path.is_absolute():
            # get the absolute path
            path = (ROOT / path).resolve()
        data_dict["path"] = str(path)

        for k in "train", "val", "test":
            if data_dict.get(k):
                x = (path / data_dict.get(k)).resolve()
                data_dict[k] = str(x)

        return data_dict

    def do_trian(self):
        ckpt = self.sutup_model()
        self.model = self.model.to(self.device)




if __name__ == '__main__':
    path = Path("./configs/dataset.yaml")
    print(path.resolve(), type(path.resolve()))
    a = yaml_load(path)
    print(a.get("names"),"\n", type(a.get("names")))


