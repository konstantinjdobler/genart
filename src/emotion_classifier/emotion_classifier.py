from torch import nn
import torch
from torchvision import models
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import wandb
from common.helpers import push_file_to_wandb
from typing import List


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

    def __init__(self, num_classes: int, label_names: List[str], lr: float, pred_threshold: float = 0.5):
        super().__init__()
        self.save_hyperparameters()

        self.model = EmotionResnetClassifier(num_classes)
        self.criterion = nn.BCEWithLogitsLoss()

    def set_argparse_config(self, config):
        '''Call before training start'''
        self.argparse_config = config
        return self

    @staticmethod
    def cross_entropy(pred, soft_targets):
        logsoftmax = nn.LogSoftmax(dim=1)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self.model(imgs)
        loss = self.criterion(outputs, labels)
        self.log("train/loss", loss, on_epoch=True,
                 on_step=False, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        outputs = self.model(imgs)
        loss = self.criterion(outputs, labels)
        self.log("val/loss", loss, on_epoch=True,
                 on_step=False, logger=True, prog_bar=True)

        p = labels.sum(dim=0)
        pred = torch.sigmoid(outputs) >= self.hparams.pred_threshold
        tp = torch.logical_and(pred == 1, labels == 1).sum(dim=0)
        fp = torch.logical_and(pred == 1, labels == 0).sum(dim=0)
        fn = torch.logical_and(pred == 0, labels == 1).sum(dim=0)
        tn = torch.logical_and(pred == 0, labels == 0).sum(dim=0)
        return {"p": p.cpu(), "tp": tp.cpu(), "fp": fp.cpu(), "fn": fn.cpu(), "tn": tn.cpu()}

    def validation_epoch_end(self, outputs):
        p, tp, fp, fn, tn = torch.zeros(self.hparams.num_classes), torch.zeros(self.hparams.num_classes), torch.zeros(
            self.hparams.num_classes), torch.zeros(self.hparams.num_classes), torch.zeros(self.hparams.num_classes)
        for v in outputs:
            p += v["p"]
            tp += v["tp"]
            fp += v["fp"]
            fn += v["fn"]
            tn += v["tn"]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        metrics = {"precision": precision, "recall": recall, "f1": f1,
                   "accuracy": accuracy, "p": p, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

        for i, label_name in enumerate(self.hparams.label_names):
            for metric_name, metric in metrics.items():
                self.log(f"{label_name}/{metric_name}", metric[i])

    def configure_optimizers(self):
        # Observe that all parameters are being optimized
        optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    def on_epoch_end(self):
        # Save preliminary model to wandb in case of crash
        push_file_to_wandb(f"{self.argparse_config.results_dir}/last.ckpt")
