import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path


def on_pretrain_routine_start(trainer):
    """Called before the pretraining routine starts."""
    pass


def on_pretrain_routine_end(trainer):
    """Called after the pretraining routine ends."""
    pass


def on_train_start(trainer):
    """Called when the training starts."""
    pass


def on_train_epoch_start(trainer):
    """Called at the start of each training epoch."""
    pass


def on_train_batch_start(trainer):
    """Called at the start of each training batch."""
    pass


def optimizer_step(trainer):
    """Called when the optimizer takes a step."""
    pass


def on_before_zero_grad(trainer):
    """Called before the gradients are set to zero."""
    pass


default_callbacks = {
    # Run in trainer
    'on_pretrain_routine_start': [on_pretrain_routine_start],
    'on_pretrain_routine_end': [on_pretrain_routine_end],
    'on_train_start': [on_train_start],
    'on_train_epoch_start': [on_train_epoch_start],
    'on_train_batch_start': [on_train_batch_start],
    'optimizer_step': [optimizer_step]}



def get_default_callbacks(x):
    """
    Return a copy of the default_callbacks dictionary with lists as default values.

    Returns:
        (defaultdict): A defaultdict with keys from default_callbacks and empty lists as default values.
    """
    return defaultdict(list, deepcopy(default_callbacks))

# get_default_callbacks(default_callbacks)
from pathlib import Path
# print(type(default_callbacks["on_train_start"]))
# print(type(default_callbacks.values()))

# s = "jsdipjgnls"
#
# a = s.startswith("nihao/jao/joaj")
# print(s, a)
# model = s
# k = all(x not in model for x in './\\')
# print(k)
# file = "helloworld"
# for f in file if isinstance(file, (list, tuple)) else [file]:
#     print(1)
# import glob
# # FILE = Path(__file__).resolve()
# # print(FILE.parents[0], FILE.parents[1], FILE.parents[2])
# ROOT = Path("/home/youtian/Documents/pro/pyCode/easy_YOLOv8/ultralytics")
# file = "VOC.yaml"
# files = glob.glob(str(ROOT / 'cfg' / '**' / file), recursive=True)
# print(files)

# import contextlib
# from copy import deepcopy
# from pathlib import Path
#
# import torch
# import torch.nn as nn
# import logging
# print(os.getenv("RANK"))
# a = 1
# set_logging(LOGGING_NAME, verbose=VERBOSE)  # run before defining LOGGER
# LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)
# LOGGER.info('kjhbn')
#
# i = 2
# # l = [i for i in range(10) if i ==1]
# for i in range(10) if i ==1 else range(20):
#     print(i)
import cv2
# import torch
# print(torch.cuda.is_available())
# print(l)

# cap = cv2.VideoCapture(-1)
# while cap.isOpened():
#     _, im = cap.read()
#     cv2.imshow("shit", im)
#     if cv2.waitKey(1) == ord('q'):
#         break
# cap.release()
#


# # this is test
# a = 3
# import os
# # print(sys.path[0]+ "ll")
# # print(os.mkdir("./kk"))
# # print(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
# assert a>5
# print(a)