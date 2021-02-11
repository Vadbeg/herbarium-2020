"""Module with configparser"""

import random
import configparser

import numpy as np
import torch


def load_config(config_path: str) -> configparser.ConfigParser:
    """
    Loads config from give path
    :param config_path: path to the config
    :return: config class
    """

    config = configparser.ConfigParser()

    config.read(filenames=config_path)

    return config


# def set_seed(seed_number: int):
#     """
#     Sets random seed for random, numpy and torch
#
#     :param seed_number: number of the seed
#     """
#
#     np.random.seed(seed_number)
#     random.seed(seed_number)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.manual_seed(seed_number)
#     torch.cuda.manual_seed_all(seed_number)
#     torch.set_printoptions(precision=10)
