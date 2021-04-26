from torch import nn
import torch
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import wandb
from common.helpers import push_file_to_wandb


class EmotionResnetClassifier(nn.Module):

    def __init__(self, num_classes: int):
        super(EmotionResnetClassifier, self).__init__()
        self.num_classes = num_classes

        self.model_ft = models.resnet18(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.model_ft.fc = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        # attr = attr.view(-1, self.num_features, 1, 1)
        # x = torch.cat([x, attr], 1)
        return self.model_ft(x)


class EmotionClassifier(pl.LightningModule):

    def __init__(self, num_classes: int):
        super().__init__()
        self.save_hyperparameters()

        self.model = EmotionResnetClassifier(num_classes)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.cross_entropy

    def set_argparse_config(self, config):
        '''Call before training start'''
        self.argparse_config = config
        return self

    @staticmethod
    def cross_entropy(pred, soft_targets):
        logsoftmax = nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

    def training_step(self, batch, batch_idx):
        imgs, features = batch
        # print("Features:", features)
        outputs = self.model(imgs)
        loss = self.criterion(outputs, features)
        self.log("train/loss", loss, on_epoch=True,
                 on_step=False, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self.model(imgs)
        loss = self.criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        _, highest_labels = torch.max(labels, 1)
        acc = torch.sum(preds == highest_labels) / labels.size(0)
        self.log("val/loss", loss, on_epoch=True,
                 on_step=False, logger=True, prog_bar=True)
        self.log("val/accuracy", acc, on_epoch=True,
                 on_step=False, logger=True, prog_bar=True)
        return loss, acc

    def configure_optimizers(self):
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1)
        return [optimizer], [exp_lr_scheduler]

    def on_epoch_end(self):
        # Save preliminary model to wandb in case of crash
        push_file_to_wandb(f"{self.argparse_config.results_dir}/last.ckpt")
