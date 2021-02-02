import json

from modules.utils import load_config
from modules.data.dataset_file_creating import transform_coco_to_dataframe


if __name__ == '__main__':
    config = load_config(config_path='/home/vadbeg/Projects/Kaggle/herbarium-2020/config.ini')

    train_metadata_path = config.get('Data', 'train_metadata_path')

    with open(file=train_metadata_path, mode='r', encoding='UTF-8', errors='ignore') as file:
        train_metadata = json.load(file)

    dataframe = transform_coco_to_dataframe(coco_metadata=train_metadata, is_valid_dataset=False)
    dataframe.to_csv(config.get('Data', 'train_dataframe_path'))
