import torch
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, DenseNet121_Weights, AlexNet_Weights, VGG16_Weights, VGG19_Weights, Inception_V3_Weights, GoogLeNet_Weights, MobileNet_V3_Small_Weights


def get_model(name="vgg16", pretrained=True):
    if name == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT
                                if pretrained else None)
    elif name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT
                                if pretrained else None)
    elif name == "densenet121":
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT
                                   if pretrained else None)
    elif name == "alexnet":
        model = models.alexnet(weights=AlexNet_Weights.DEFAULT
                               if pretrained else None)
    elif name == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.DEFAULT
                             if pretrained else None)
    elif name == "vgg19":
        model = models.vgg19(weights=VGG19_Weights.DEFAULT
                             if pretrained else None)
    elif name == "inception_v3":
        model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT
                                    if pretrained else None)
    elif name == "googlenet":
        model = models.googlenet(weights=GoogLeNet_Weights.DEFAULT
                                 if pretrained else None)
    elif name == 'mobilenet':
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT
                                          if pretrained else None)

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model
