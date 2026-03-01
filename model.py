import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet152_Weights

class ColorizationNet(nn.Module):
    def __init__(self, midlevel_input_size=512, global_input_size=512):
        super(ColorizationNet, self).__init__()
        self.midlevel_input_size = midlevel_input_size
        self.global_input_size = global_input_size

        self.conv1 = nn.Conv2d(midlevel_input_size, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, midlevel_input):
        # Convolutional layers and upsampling
        x = F.relu(self.bn2(self.conv1(midlevel_input)))
        x = self.upsample(x)
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = F.relu(self.conv4(x))
        x = self.upsample(self.conv5(x))
        return x


class ColorNet(nn.Module):
    def __init__(self, use_pretrained=True):
        super(ColorNet, self).__init__()

        # Build ResNet and change first conv layer to accept single-channel input
        weights = ResNet152_Weights.DEFAULT if use_pretrained else None
        resnet_model = models.resnet152(weights=weights)
        
        # Modify the first layer to accept 1 channel instead of 3 (summing the pretrained 3-channel weights)
        resnet_model.conv1.weight = nn.Parameter(resnet_model.conv1.weight.sum(dim=1).unsqueeze(1).data)

        # Extract midlevel features from resnet_model
        # First 6 layers are selected (gives 512 channels out)
        self.midlevel_resnet = nn.Sequential(*list(resnet_model.children())[0:6])
        self.fusion_and_colorization_net = ColorizationNet()

    def forward(self, input_image):
        # Pass input through resnet_model to extract features
        midlevel_output = self.midlevel_resnet(input_image)

        # Combine features in fusion layer and upsample
        output = self.fusion_and_colorization_net(midlevel_output)
        return output
