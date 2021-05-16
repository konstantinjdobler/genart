from pathlib import Path
from typing import Callable, List, Optional
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

    def __init__(self, data_dir: str, batch_size: int, num_workers: int, image_resizing: int, fast_debug: bool = False):
        super().__init__()
        # check if we can use dataset resized to smaller size
        available_resizes = [dir[0].split("-")[-1]
                             for dir in os.walk(data_dir) if "images-" in dir[0]]
        resize_suffix = "" if len(
            available_resizes) == 0 else f"-{min((int(size) for size in available_resizes if int(size) >= image_resizing))}"
        self.image_subfolder = Path(data_dir + f"/images{resize_suffix}")
        self.annotation_path = Path(
            data_dir + "/WikiArt-Emotions/WikiArt-Emotions-Ag4.tsv")
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

    def prepare_data(self):
        self.train_set = AnnotatedImageDataset(str(self.image_subfolder), str(
            self.annotation_path), transform=self.transform, fast_debug=self.fast_debug)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)


ID_COLUMN = "ID"
PATH_COLUMN = "_path"
FilterColumnFunction = Callable[[str, int], bool]


class WikiArtEmotionsDataModule2(pl.LightningDataModule):

    def __init__(self, image_folder: str, annotations_file: str, resize_size: int, crop_size: int, batch_size: int, num_workers: int, filter_columns_function: Optional[FilterColumnFunction] = None, fast_debug: bool = False):
        super().__init__()

        self.image_folder = image_folder
        self.annotations_file = annotations_file
        df = pd.read_csv(self.annotations_file, "\t")

        if filter_columns_function:
            filtered_columns = [c for idx, c in enumerate(
                df.columns) if filter_columns_function(c, idx) or c == ID_COLUMN]
            df = df.filter(items=filtered_columns)

        df[PATH_COLUMN] = df.apply(
            lambda x: f"{image_folder}/{x[ID_COLUMN]}.jpg", axis=1)
        self.df_train, self.df_val = train_test_split(df, train_size=0.8)

        self.train_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(crop_size),
            transforms.Resize(resize_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fast_debug = fast_debug

    def prepare_data(self):
        self.train_data = AnnotatedImageDataset2(
            data=self.df_train, transforms=self.train_transforms, fast_debug=self.fast_debug)
        self.val_data = AnnotatedImageDataset2(
            data=self.df_val, transforms=self.val_transforms, fast_debug=self.fast_debug)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)


class AnnotatedImageDataset2(Dataset):

    def __init__(self, data: pd.DataFrame, transforms, fast_debug: bool = False, meta_data_columns: List[str] = [ID_COLUMN, PATH_COLUMN]):
        super().__init__()
        self.data = data
        if fast_debug:
            self.data = data[:20]
        self.meta_data = self.data[meta_data_columns]
        self.labels = self.data.drop(meta_data_columns, axis=1)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_data = self.meta_data.iloc[index]

        img = Image.open(img_data["_path"])
        if self.transforms:
            img = self.transforms(img)
        img_labels = self.labels.iloc[index]

        return img, img_labels.values.astype(float)


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

    def prepare_data(self):
        self.train_set = CelebAImageFeatureFolder(str(self.image_subfolder), str(
            self.annotation_path), transform=self.transform, fast_debug=self.fast_debug)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
