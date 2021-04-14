from pathlib import Path
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from torchvision.transforms.transforms import Resize


class WikiArtEmotionsDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, image_resizing: int, fast_debug: bool = False):
        super().__init__()
        self.image_subfolder = Path(data_dir + "/images")
        self.annotation_path = Path(
            data_dir + "/WikiArt-Emotions/WikiArt-Emotions-Ag4.tsv")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_resizing
        self.fast_debug = fast_debug

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(self.image_size),
            transforms.RandomCrop(self.image_size)
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, self.image_size, self.image_size)
        # self.num_classes = 10

    def prepare_data(self):
        self.train_set = AnnotatedImageDataset(str(self.image_subfolder), str(
            self.annotation_path), transform=self.transform, fast_debug=self.fast_debug)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)


class AnnotatedImageDataset(Dataset):
    def __init__(self, image_root, annotation_file, transform, fast_debug):
        super(AnnotatedImageDataset, self).__init__()
        self.transform = transform

        with open(annotation_file, 'r') as f:
            data = f.read()
        data = data.strip().split('\n')

        self.image_files = [
            {
                'path': Path(image_root) / (entry.split('\t')[0] + ".jpg"),
                'annotations': torch.FloatTensor(list(map(float, entry.split('\t')[12:])))
            } for entry in data[1:]
        ]
        if fast_debug:
            self.image_files = self.image_files[:20]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_data = self.image_files[index]

        img = Image.open(str(img_data['path']))
        if self.transform:
            img = self.transform(img)
        return img, img_data['annotations']
