"""Module with network code"""

import timm
import torch


class HerbariumNet(torch.nn.Module):
    """Neural network class"""

    def __init__(self, model_type: str, pretrained: bool = True, num_of_output_nodes: int = 1000):
        """
        Init class method

        :param model_type: model type (resnet18, effnet)
        :param pretrained: if True uses pretrained weights for network
        :param num_of_output_nodes: number of output network nodes
        """

        super().__init__()

        backbone = timm.create_model(model_type, pretrained=pretrained)

        if 'resnet' in model_type:
            n_features = backbone.fc.in_features
        elif 'eff' in model_type:
            n_features = backbone.classifier.in_features
        else:
            raise NotImplementedError(f'No num_features for this network type')

        self.backbone = torch.nn.Sequential(*backbone.children())[:-2]
        self.classifier = torch.nn.Linear(n_features, num_of_output_nodes)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x: torch.tensor) -> torch.tensor:
        """
        Performs forward propagation to get features from backbone

        :param x: input tensor
        :return: features tensor
        """

        x = self.backbone(x)

        return x

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Performs forward pass threw network

        :param x: input tensor
        :return: network result
        """

        feats = self.forward_features(x)

        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)

        return x
