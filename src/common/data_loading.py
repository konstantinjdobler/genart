from pathlib import Path
from typing import Callable, List, Optional, OrderedDict
import torch
import pytorch_lightning as pl
from torch.utils import data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,  Dataset
from PIL import Image
import os
# from sklearn.model_selection import train_test_split
import torchvision
import pandas as pd

WIKIART_EMOTIONS_MEANS = [0.50095144, 0.44335716, 0.38643128]  # of production
WIKIART_EMOTIONS_STDS = [0.22497191,
                         0.21211041, 0.19994931]  # better get tested


class WikiArtEmotionsDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, annotation_path: str, batch_size: int, num_workers: int, image_resizing: int, fast_debug: bool = False):
        super().__init__()
        # check if we can use dataset resized to smaller size
        available_resizes = [dir[0].split("-")[-1]
                             for dir in os.walk(data_dir) if "images-" in dir[0]]
        possible_resizes = [
            int(size) for size in available_resizes if int(size) >= image_resizing]
        resize_suffix = "" if len(
            possible_resizes) == 0 else f"-{min(possible_resizes)}"
        self.image_subfolder = Path(data_dir + f"/images{resize_suffix}")
        self.annotation_path = annotation_path
        print("Using dataset", self.image_subfolder,
              "and annotation file", self.annotation_path)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_resizing
        self.fast_debug = fast_debug

        # TODO: I think we DON'T want to use the actual data stats here, because that would not normalize to [-1,1]. Am I correct?
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(self.image_size),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip()
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, self.image_size, self.image_size)
        # self.num_classes = 10

    def setup(self, stage):
        self.train_set = AnnotatedImageDataset(str(self.image_subfolder), str(
            self.annotation_path), transform=self.transform, fast_debug=self.fast_debug)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


class AnnotatedImageDataset(Dataset):
    def __init__(self, image_root, annotation_file, transform, fast_debug):
        super(AnnotatedImageDataset, self).__init__()
        self.transform = transform

        with open(annotation_file, 'r') as f:
            data = f.read()
        data = data.strip().split('\n')

        # Images get normalized to [-1,1], so we want our labels to be in the same value range
        annotation_map = {"1": 1.0, "0": -1.0}

        self.image_files = [
            {
                'path': Path(image_root) / (entry.split('\t')[0] + ".jpg"),
                'annotations': torch.FloatTensor(list(map(annotation_map.get, entry.split('\t')[32:52])))
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


class CSChanDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, annotation_path: str, batch_size: int, num_workers: int, image_resizing: int, fast_debug: bool = False):
        super().__init__()
        # check if we can use dataset resized to smaller size
        available_resizes = [dir[0].split("-")[-1].split("/")[0]
                             for dir in os.walk(data_dir) if "images-" in dir[0]]
        possible_resizes = [
            int(size) for size in available_resizes if int(size) >= image_resizing]
        resize_suffix = "" if len(
            possible_resizes) == 0 else f"-{min(possible_resizes)}"
        self.image_subfolder = Path(data_dir + f"/images{resize_suffix}")
        self.annotation_path = annotation_path
        print("Using dataset", self.image_subfolder,
              "and annotation file", self.annotation_path)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_resizing
        self.fast_debug = fast_debug

        # TODO: I think we DON'T want to use the actual data stats here, because that would not normalize to [-1,1]. Am I correct?
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip()
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (3, self.image_size, self.image_size)
        # self.num_classes = 10

    def setup(self, stage):
        self.train_set = CSChanImageDataset(str(self.image_subfolder), str(
            self.annotation_path), transform=self.transform, fast_debug=self.fast_debug)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


class CSChanImageDataset(Dataset):
    def __init__(self, image_root, annotation_file, transform, fast_debug):
        super(CSChanImageDataset, self).__init__()
        self.transform = transform

        with open(annotation_file, 'r') as f:
            data = f.read()
        data = data.strip().split('\n')

        # Images get normalized to [-1,1], so we want our labels to be in the same value range
        self.image_files = []
        for entry in data:
            annotation_map = OrderedDict([("Abstract_Expressionism", -1.0), ("Action_painting", -1.0), ("Analytical_Cubism", -1.0), ("Art_Nouveau_Modern", -1.0), ("Baroque", -1.0), ("Color_Field_Painting", -1.0), ("Contemporary_Realism", -1.0), ("Cubism", -1.0), ("Early_Renaissance", -1.0), ("Expressionism", -1.0), ("Fauvism", -1.0), ("High_Renaissance", -1.0), ("Impressionism", -1.0),
                                         ("Mannerism_Late_Renaissance", -1.0), ("Minimalism", -1.0), ("Naive_Art_Primitivism", -1.0), ("New_Realism", -1.0), ("Northern_Renaissance", -1.0), ("Pointillism", -1.0), ("Pop_Art", -1.0), ("Post_Impressionism", -1.0), ("Realism", -1.0), ("Rococo", -1.0), ("Romanticism", -1.0), ("Symbolism", -1.0), ("Synthetic_Cubism", -1.0), ("Ukiyo_e", -1.0)])
            if float(entry.split(',')[1]) == 6:
                annotation_map[entry.split('/')[0]] = 1.0
                self.image_files.append({
                    'path': Path(image_root) / (entry.split(',')[0]),
                    'annotations': torch.FloatTensor(list(annotation_map.values()))
                })

        if fast_debug:
            self.image_files = self.image_files[: 20]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_data = self.image_files[index]

        img = Image.open(img_data['path'])
        if self.transform:
            img = self.transform(img)
        return img, img_data['annotations']


class CelebAImageFeatureFolder(torchvision.datasets.ImageFolder):
    def __init__(self, image_root, landmark_file, transform, fast_debug):
        super(CelebAImageFeatureFolder, self).__init__(
            root=image_root, transform=transform)

        with open(landmark_file, 'r') as f:
            data = f.read()
        data = data.strip().split('\n')
        self.attrs = torch.FloatTensor(
            [list(map(float, line.split()[1:])) for line in data[2:]])
        if fast_debug:
            self.attrs = self.attrs[:20]
            self.imgs = self.imgs[:20]
            self.samples = self.imgs

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)

        return img, self.attrs[index]


class CelebADataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, image_resizing: int, fast_debug: bool = False):
        super().__init__()
        # check if we can use dataset resized to smaller size
        # available_resizes = [dir[0].split("-")[-1]
        #                      for dir in os.walk(data_dir) if "images-" in dir[0]]
        # resize_suffix = "" if len(
        #     available_resizes) == 0 else f"-{min((size for size in available_resizes if int(size) >= image_resizing))}"
        self.image_subfolder = Path(data_dir)
        self.annotation_path = Path(
            data_dir + "/landmark.txt")
        print("Using dataset", self.image_subfolder,
              "and annotation file", self.annotation_path)
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

    def setup(self, stage):
        self.train_set = CelebAImageFeatureFolder(str(self.image_subfolder), str(
            self.annotation_path), transform=self.transform, fast_debug=self.fast_debug)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
