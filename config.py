import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_optimizer import Lookahead

device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = "checkpoints/"
num_workers = 16


def get_loss():
    return nn.CrossEntropyLoss(label_smoothing=0.1)

# Gradient clipping value
grad_clip_val = 1.0

class CNNConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    num_epochs = 2
    batch_size = 32
    device = device
    out_name = "cnn_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.Adam(model.parameters(), lr=CNNConfig.learning_rate)


class VGGConfig:
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    num_epochs = 2
    learning_rate = 1e-4
    device = device
    out_name = "vgg16_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.Adam(model.parameters(), lr=VGGConfig.learning_rate)


class EfficientConfig:
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4
    device = device
    out_name = save_dir + "EfficientNet/efficientnetb0_finetune_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.Adam(model.parameters(), lr=EfficientConfig.learning_rate)


class ViTConfig:
    criterion = nn.CrossEntropyLoss()
    num_epochs = 100
    batch_size = 64
    learning_rate = 3e-4
    weight_decay = 0.01
    T_max = 50
    device = device
    out_name = save_dir + "VisionTransfomers/ViT_finetune_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.AdamW(model.parameters(), lr=ViTConfig.learning_rate)

    @staticmethod
    def scheduler(optimizer):
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, 
            T_max=ViTConfig.T_max
        )


class ResNetConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    weight_decay = 1e-4
    num_epochs = 100
    batch_size = 32
    device = device
    out_name = save_dir +"ResNet50/resnet50_model"
    num_workers = num_workers

    @staticmethod
    def optimizer_fn(model):
        return optim.AdamW(
            model.parameters(),
            lr=ResNetConfig.learning_rate,
            weight_decay=ResNetConfig.weight_decay
        )

    @staticmethod
    def scheduler_fn(optimizer):
        return CosineAnnealingLR(optimizer, T_max=ResNetConfig.num_epochs)


class DenseNetConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    weight_decay = 1e-4
    num_epochs = 100
    batch_size = 32
    device = device
    out_name = save_dir +"DenseNet/densenet121_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.AdamW(
            model.parameters(),
            lr=DenseNetConfig.learning_rate,
            weight_decay=DenseNetConfig.weight_decay,
        )


class ConvNeXtConfig:
    criterion = get_loss()
    learning_rate = 3e-4
    num_epochs = 100
    batch_size = 32
    device = device
    out_name = save_dir + "ConvNeXt/convnext_tiny_model"
    grad_clip = grad_clip_val
    @staticmethod
    def optimizer_fn(model):
        base_opt = optim.AdamW(model.parameters(), lr=ConvNeXtConfig.learning_rate, weight_decay=1e-4)
        return Lookahead(base_opt)


class SwinConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 100
    batch_size = 32
    device = device
    out_name = save_dir +"SwinTransformer/swin_transformer_tiny_model"
    model_name = "microsoft/swin-tiny-patch4-window7-224"

    @staticmethod
    def optimizer_fn(model):
        return optim.AdamW(model.parameters(), lr=SwinConfig.learning_rate, weight_decay=SwinConfig.weight_decay)
