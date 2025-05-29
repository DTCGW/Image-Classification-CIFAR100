import torch
import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = "checkpoints/"

class CNNConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    num_epochs = 2
    batch_size = 32
    device = device
    out_name = save_dir+"CNN/cnn_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.Adam(model.parameters(), lr=CNNConfig.learning_rate)


class VGGConfig:
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    num_epochs = 2
    learning_rate = 1e-4
    device = device
    out_name = save_dir+"vgg16/vgg16_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.Adam(model.parameters(), lr=VGGConfig.learning_rate)


class EfficientConfig:
    criterion = nn.CrossEntropyLoss()
    batch_size = 32
    num_epochs = 2
    learning_rate = 1e-4
    device = device
    out_name = save_dir+"Efficient/efficientnetb0_finetune_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.Adam(model.parameters(), lr=EfficientConfig.learning_rate)


class ViTConfig:
    criterion = nn.CrossEntropyLoss()
    num_epochs = 2
    batch_size = 32
    learning_rate = 2e-5
    weight_decay = 0.01
    T_max = 10
    device = device
    out_name = save_dir+"ViT/ViT_finetune_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.AdamW(model.parameters(), lr=ViTConfig.learning_rate)

    @staticmethod
    def scheduler():
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer=ViTConfig.optimizer_fn, 
            T_max=ViTConfig.T_max
        )


class ResNetConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    num_epochs = 10
    batch_size = 32
    device = device
    out_name = save_dir +"ResNet50/resnet50_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.Adam(model.parameters(), lr=ResNetConfig.learning_rate)


class DenseNetConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-3
    num_epochs = 10
    batch_size = 32
    device = device
    out_name = save_dir+"DenseNet/densenet121_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.Adam(model.parameters(), lr=DenseNetConfig.learning_rate)


class ConvNeXtConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    num_epochs = 10
    batch_size = 32
    device = device
    out_name = save_dir+"ConvNeXt/convnext_tiny_model"

    @staticmethod
    def optimizer_fn(model):
        return optim.AdamW(model.parameters(), lr=ConvNeXtConfig.learning_rate)


class SwinConfig:
    criterion = nn.CrossEntropyLoss()
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 10
    batch_size = 32
    device = device
    out_name = save_dir+"swin_transformer_tiny_model"
    model_name = "Swin/swin-tiny-patch4-window7-224"

    @staticmethod
    def optimizer_fn(model):
        return optim.AdamW(model.parameters(), lr=SwinConfig.learning_rate, weight_decay=SwinConfig.weight_decay)
