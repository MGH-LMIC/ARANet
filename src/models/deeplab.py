import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
num_classes = 3

class DeepLabWrapper(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabWrapper, self).__init__()
        self.model = deeplabv3_resnet50(pretrained = True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)['out']
