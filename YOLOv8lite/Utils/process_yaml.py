import yaml
from pathlib import Path
from typing import Union


def yaml_load(file:Union[Path, str]):

    file = str(file) if isinstance(file, Path) else file
    with open(file, encoding='utf-8') as f:

        cfg_list = f.read()
        cfg_dict = {**yaml.safe_load(cfg_list)}

        return cfg_dict