import torch, torchvision

from torchvision import models
from torch import nn




class ImageNetNormalization(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super(ImageNetNormalization, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return torchvision.transforms.functional.normalize(x, self.mean, self.std)
    

def load_resnet(MODEL_PATH):
    base_resnet = models.resnet18(pretrained=True)

    num_ftrs_in = base_resnet.fc.in_features
    num_ftrs_out = 10
    base_resnet.fc = nn.Linear(num_ftrs_in, num_ftrs_out)
    resnet = torch.nn.Sequential(
        ImageNetNormalization(),
        base_resnet
    )

    state_dict = torch.load(MODEL_PATH)
    resnet.load_state_dict(state_dict)
    resnet = resnet
    model = resnet
    model.eval()
    return model

def load_mobilenet(MODEL_PATH):

    base_mobilenet = models.mobilenet_v3_small(pretrained=True)

    num_ftrs_in = base_mobilenet.classifier[0].in_features
    num_ftrs_out = base_mobilenet.classifier[0].out_features
    base_mobilenet.classifier[0] = nn.Linear(num_ftrs_in, num_ftrs_out)

    num_ftrs_in = base_mobilenet.classifier[3].in_features
    num_ftrs_out = 10
    base_mobilenet.classifier[3] = nn.Linear(num_ftrs_in, num_ftrs_out)

    mobilenet = torch.nn.Sequential(
        ImageNetNormalization(),
        base_mobilenet
    )

    
    state_dict = torch.load(MODEL_PATH)
    mobilenet.load_state_dict(state_dict)
    mobilenet = mobilenet

    model = mobilenet
    model.eval()
    
    return model