import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import re
import yaml



def yaml_save(file='data.yaml', data=None):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    for k, v in data.items():
        if isinstance(v, Path):
            data[k] = str(v)

    # Dump data to file in YAML format
    with open(file, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def yaml_load(file='data.yaml', append_filename=False):
    with open(file, errors='ignore', encoding='utf-8') as f:
        # 打开yolo8.yaml,read（）是以字符串的形式输出
        s = f.read()
        # print(s)

        # Remove special characters
        if not s.isprintable():
            # print(s.isprintable())
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def find_nonprintable_chars(string):
    for char in string:
        ascii_value = ord(char)
        if ascii_value < 32:
            print(f"发现非可打印字符: {char} (ASCII值: {ascii_value})")


if __name__ == '__main__':

    # ya = yaml_load(file="/home/youtian/Documents/pro/pyCode/easy_YOLOv8/ultralytics/cfg/models/v8/yolov8.yaml")
    # s = "**##H\tello \n ####@ World!"
    # print(find_nonprintable_chars(s))
    # # print(s.isprintable())
    # s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
    # print(ya.get("activation", 1))
    # print(ya)
    # a = [3, 4, 1, 9]
    # print(type(*a[1:]))


    # pattren = "s"
    k = "s"
    res = re.match(k, "sdfgas")
    print(res)
    print(res.group())