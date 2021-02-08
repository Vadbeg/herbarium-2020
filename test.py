"""Modules with testing"""

import json
import random

import torch
import pandas as pd
import matplotlib.pyplot as plt

from modules.data.dataset import ImageDataset
from modules.data.augs import train_augmentations, valid_augmentations
from modules.data.utils import tensor_to_numpy, numpy_to_tensor
from modules.model.utils import load_model
from modules.utils import load_config

if __name__ == '__main__':
    config_path = '/home/vadbeg/Projects/Kaggle/herbarium-2020/config.ini'
    config = load_config(config_path=config_path)

    train_images_path = config['Data']['train_images_folder']
    train_dataframe_path = config['Data']['train_dataframe_path']

    image_size = tuple(json.loads(config.get('Model', 'image_size')))

    train_dataframe = pd.read_csv(train_dataframe_path)

    image_dataset = ImageDataset(dataframe=train_dataframe,
                                 images_path=train_images_path,
                                 augmentations=train_augmentations,
                                 image_size=image_size)

    RANDOM_IDX = random.randint(0, len(image_dataset) - 1)

    image_dataset_item = image_dataset[RANDOM_IDX]
    image = image_dataset_item['image']
    image = torch.unsqueeze(image, dim=0)

    model = load_model(model_path='/home/vadbeg/Projects/Kaggle/herbarium-2020/weights/02-02-2021-19:24:46.pt')
    result: torch.Tensor = model(image)[0]
    result = torch.softmax(result, dim=0)

    result_index = torch.argmax(result.squeeze(0), dim=0).data
    result_confidence = torch.max(result.squeeze(0), dim=0).values.data

    image = image.squeeze(0)
    image = tensor_to_numpy(image)

    plt.title(f'{result_index} -- {result_confidence:.2f}')

    plt.imshow(image)
    plt.show()
