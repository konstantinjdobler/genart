from pathlib import Path
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split


class WikiArtEmotionsDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, image_resizing: int):
        super().__init__()
        self.image_subfolder = Path(data_dir + "/images")
        self.annotation_path = Path(
            data_dir + "/WikiArt-Emotions/WikiArt-Emotions-Ag4.tsv")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_resizing

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomCrop(self.image_size)
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, self.image_size, self.image_size)
        # self.num_classes = 10

    def setup(self, stage=None):
        self.train_set = ImageFeatureFolder(str(self.data_dir), str(
            self.annotation_path), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)


class ImageFeatureFolder(datasets.ImageFolder):
    def __init__(self, image_root, annotation_file, transform):
        super(ImageFeatureFolder, self).__init__(
            root=image_root, transform=transform)

        with open(annotation_file, 'r') as f:
            data = f.read()
        data = data.strip().split('\n')
        self.attrs = torch.FloatTensor(
            [list(map(float, line.split('\t')[12:])) for line in data[1:]])

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)

        return img, self.attrs[index]
