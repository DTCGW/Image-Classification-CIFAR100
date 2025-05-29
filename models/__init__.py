from .CNN import BasicCNN, CNNProcessor
from .VGG16 import VGG16_NET, VGGProcessor
from .Efficient import EFFICIENT_B0
from .ViT import VisionTransfomers
from .SVM import SVM
from .ResNet import ResNet50_NET, train_model as train_model_resnet, evaluate as evaluate_resnet
from .DenseNet import DenseNet121_NET, train_model as train_model_densenet, evaluate as evaluate_densenet
from .ConvNeXt import ConvNeXtTiny_NET, train_model as train_model_convnext, evaluate as evaluate_convnext
from .SwinTransformer import build_model as build_model, train_model as train_model_swinTransformer, evaluate as evaluate_swinTransformer
