

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.models as Model
import os
import torchvision.models as Model

class TwoStreamMobileNetV3L(nn.Module):
  def __init__(self, model_path1, model_path2, num_classes = 10):
        super(TwoStreamMobileNetV3L, self).__init__()

        # Load the pretrained models
        self.model1 = Model.mobilenet_v3_large()
        num_feature = self.model1.classifier[0].in_features
        output_shape = num_classes

        self.model1.classifier = nn.Sequential(
            nn.Linear(in_features = num_feature, out_features = 1280),
            nn.Hardswish(),
            nn.Dropout(0.5),
            nn.Linear(1280, output_shape)
        )


        self.model2 = Model.mobilenet_v3_large()
        self.model2.classifier = nn.Sequential(
            nn.Linear(in_features = num_feature, out_features = 1280),
            nn.Hardswish(),
            nn.Dropout(0.5),
            nn.Linear(1280, output_shape)
        )

        self.model1.load_state_dict(model_path1['state_dict'])
        self.model2.load_state_dict(model_path2['state_dict'])

        # Freeze the parameters
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False


        # New classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_feature*2, 1280),
            nn.Hardswish(),
            nn.Dropout(0.5),
            nn.Linear(1280, num_classes)
        )

  def forward(self, x1, x2):
    # Extract features from both models without passing through their classifiers
    x1 = self.model1.features(x1)
    x1 = self.model1.avgpool(x1)
    x1 = torch.flatten(x1, 1)

    x2 = self.model2.features(x2)
    x2 = self.model2.avgpool(x2)
    x2 = torch.flatten(x2, 1)

    x = torch.cat((x1, x2), dim=1)  # Concatenate along the feature dimension
    x = self.classifier(x)
    return x
