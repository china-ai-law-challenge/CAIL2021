"""Utils tools.

Author: Yixu GAO yxgao19@fudan.edu.cn
"""
import logging
import os
from collections import OrderedDict

import torch


def get_path(path):
    """Create the path if it does not exist.

    Args:
        path: path to be used

    Returns:
        Existed path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_csv_logger(log_file_name,
                   title='',
                   log_format='%(message)s',
                   log_level=logging.INFO):
    """Get csv logger.

    Args:
        log_file_name: file name
        title: first line in file
        log_format: default: '%(message)s'
        log_level: default: logging.INFO

    Returns:
        csv logger
    """
    logger = logging.getLogger(log_file_name)
    logger.setLevel(log_level)
    file_handler = logging.FileHandler(log_file_name, 'w')
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    if title:
        logger.info(title)
    return logger


def load_torch_model(model, model_path):
    """Load state dict to model.

    Args:
        model: model to be loaded
        model_path: state dict file path

    Returns:
        loaded model
    """
    pretrained_model_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, value in pretrained_model_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict, strict=True)
    return model
