"""Module with network training"""

import os
import sys
from datetime import datetime
from typing import List, Dict

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from modules.model.metrics import get_accuracy, get_f1_score
from modules.model.utils import print_report, save_weights, save_report


def train_model(model: torch.nn.Module, num_epochs: int,
                optimizer, loss_func,
                train_dataloader: DataLoader, valid_dataloader: DataLoader,
                device: str = 'cuda:0',
                report_dir: str = 'reports',
                weights_dir: str = 'weights') -> List[Dict[str, List[float]]]:
    """
    Performs whole training process

    :param model: model to train
    :param train_dataloader: train dataloader
    :param valid_dataloader: valid dataloader
    :param optimizer: optimizer used
    :param loss_func: loss functions used
    :param device: device to calculate on
    :param report_dir: directory to save reports
    :param weights_dir: directory to save weights
    :return: list of dicts with metrics
    """

    device = torch.device(device)

    file_name = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    weights_path = os.path.join(weights_dir,
                                f'{file_name}.pkl')
    report_path = os.path.join(report_dir,
                               f'{file_name}.json')

    model.to(device=device)

    metrics_list = list()

    for epoch in range(num_epochs):
        epoch_metrics_dict = dict()

        model.train()
        train_metrics = train_one_epoch(model=model, dataloader=train_dataloader,
                                        epoch_idx=epoch, loss_func=loss_func,
                                        optimizer=optimizer, device=device)
        train_epoch_loss_list, train_epoch_accuracy_list, train_epoch_f1_list = train_metrics

        epoch_metrics_dict['train_loss'] = train_epoch_loss_list
        epoch_metrics_dict['train_accuracy'] = train_epoch_accuracy_list
        epoch_metrics_dict['train_f1'] = train_epoch_f1_list

        model.eval()
        if valid_dataloader:
            valid_metrics = valid_one_epoch(model=model, dataloader=valid_dataloader,
                                            epoch_idx=epoch, loss_func=loss_func,
                                            device=device)
            valid_epoch_loss_list, valid_epoch_accuracy_list, valid_epoch_f1_list = valid_metrics

            epoch_metrics_dict['valid_loss'] = valid_epoch_loss_list
            epoch_metrics_dict['valid_accuracy'] = valid_epoch_accuracy_list
            epoch_metrics_dict['valid_f1'] = valid_epoch_f1_list

        metrics_list.append(epoch_metrics_dict)

        print(epoch_metrics_dict)

        print_report(metrics=epoch_metrics_dict)
        save_report(report_path=report_path, metrics=epoch_metrics_dict, epoch_idx=epoch)
        save_weights(model=model, metrics_list=metrics_list, weights_path=weights_path)

    return metrics_list


def train_one_epoch(model: torch.nn.Module, dataloader: DataLoader,
                    optimizer, loss_func,
                    epoch_idx: int, device: torch.device):
    """
    Performs training phase

    :param model: model to train
    :param dataloader: train dataloader
    :param optimizer: optimizer used
    :param loss_func: loss functions used
    :param epoch_idx: index of the epoch
    :param device: device to calculate on
    :return: tuple with metrics
    """

    epoch_loss_numbers = list()
    epoch_accuracy_numbers = list()
    epoch_f1_numbers = list()

    tqdm_dataloader = tqdm(dataloader, file=sys.stdout)

    for dataset_element in tqdm_dataloader:
        optimizer.zero_grad()

        image = dataset_element['image'].to(device)
        label = dataset_element['label'].to(device)

        result = model(image)

        loss_value = loss_func(result, label)

        loss_value.backward()
        optimizer.step()

        loss_number = round(loss_value.detach().cpu().item(), 3)
        accuracy_number = round(get_accuracy(model_prediction=result, true_prediction=label), 3)
        f1_score_number = round(get_f1_score(model_prediction=result, true_prediction=label), 3)

        tqdm_dataloader.set_postfix(
            text=f'Epoch: {epoch_idx}; '
                 f'Loss number: {loss_number}; '
                 f'Accuracy number: {accuracy_number}; '
                 f'F1-score number: {f1_score_number}; '
        )

        epoch_loss_numbers.append(loss_number)
        epoch_accuracy_numbers.append(accuracy_number)
        epoch_f1_numbers.append(f1_score_number)

    return epoch_loss_numbers, epoch_accuracy_numbers, epoch_f1_numbers


def valid_one_epoch(model: torch.nn.Module, dataloader: DataLoader,
                    loss_func,
                    epoch_idx: int, device: torch.device):
    """
    Performs evaluation phase

    :param model: model to train
    :param dataloader: validation dataloader
    :param loss_func: loss functions used
    :param epoch_idx: index of the epoch
    :param device: device to calculate on
    :return: tuple with metrics
    """

    epoch_loss_numbers = list()
    epoch_accuracy_numbers = list()
    epoch_f1_numbers = list()

    with torch.no_grad():
        tqdm_dataloader = tqdm(dataloader, file=sys.stdout)

        for dataset_element in tqdm_dataloader:
            image = dataset_element['image'].to(device)
            label = dataset_element['label'].to(device)

            result = model(image)

            loss_value = loss_func(result, label)

            loss_number = round(loss_value.detach().cpu().item(), 3)
            accuracy_number = round(get_accuracy(model_prediction=result, true_prediction=label), 3)
            f1_score_number = round(get_f1_score(model_prediction=result, true_prediction=label), 3)

            tqdm_dataloader.set_postfix(
                text=f'Epoch: {epoch_idx}; '
                     f'Loss number: {loss_number}; '
                     f'Accuracy number: {accuracy_number}; '
                     f'F1-score number: {f1_score_number}; '
            )

            epoch_loss_numbers.append(loss_number)
            epoch_accuracy_numbers.append(accuracy_number)
            epoch_f1_numbers.append(f1_score_number)

    return epoch_loss_numbers, epoch_accuracy_numbers, epoch_f1_numbers

