from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

class HSTS(Dataset):
    def __init__(self, image_path, label_path):
        super().__init__()


    def __getitem__(self, item):
        ...

    def __len__(self):
        ...


 test_set= DataLoader(dataset=HSTS, batch_size=4, shuffle=True, num_workers=4, drop_last=False)