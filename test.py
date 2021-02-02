"""Modules with testing"""

import pandas as pd
import matplotlib.pyplot as plt

from modules.data.dataset import ImageDataset
from modules.data.augs import train_augmentations, valid_augmentations
from modules.data.utils import tensor_to_numpy, numpy_to_tensor
from modules.utils import load_config

if __name__ == '__main__':
    config_path = '/home/vadbeg/Projects/Kaggle/cassava-leaf-disease/config.ini'
    config = load_config(config_path=config_path)

    train_images_path = config['Data']['train_images_path']
    train_dataframe_path = config['Data']['train_dataframe_path']

    train_dataframe = pd.read_csv(train_dataframe_path)

    image_dataset = ImageDataset(dataframe=train_dataframe,
                                 images_path=train_images_path,
                                 augmentations=train_augmentations)

    res = image_dataset[0]

    print(res)

    image = res['image']

    image = tensor_to_numpy(image)

    plt.imshow(image)
    plt.show()
