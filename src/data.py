from collections import namedtuple
from typing import Tuple

import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, default_collate, random_split
from torchvision.datasets import CelebA, FashionMNIST

DataInfo = namedtuple("DataInfo", "image_channels image_size num_classes sigma_data")
DataLoaders = namedtuple("DataLoaders", "train valid")


def load_dataset_and_make_dataloaders(
    dataset_name: str,
    root_dir: str,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoaders, DataInfo]:

    train_dataset, valid_dataset, data_info = load_dataset(dataset_name, root_dir)
    dl = make_dataloaders(
        train_dataset,
        valid_dataset,
        data_info.num_classes,
        batch_size,
        num_workers,
        pin_memory,
    )
    return dl, data_info


def load_dataset(
    dataset_name="FashionMNIST", root_dir="data"
) -> Tuple[Dataset, Dataset, DataInfo]:

    if dataset_name == "FashionMNIST":
        t = T.Compose([T.ToTensor(), T.Pad(2), T.Normalize(mean=(0.5,), std=(0.5,))])
        train_dataset = FashionMNIST(root_dir, download=True, transform=t)
        train_dataset, valid_dataset = random_split(train_dataset, [50000, 10000])
        num_classes = 10
        c, h, sigma_data = 1, 32, 0.6648

    elif dataset_name == "CelebA":
        t = T.Compose(
            [
                T.ToTensor(),
                T.CenterCrop(178),
                T.Resize(128, antialias=True),
                T.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )
        # Download disabled due to issues with GDrive
        train_dataset = CelebA(root_dir, download=False, transform=t)
        train_dataset, valid_dataset = random_split(train_dataset, [150000, 12770])
        num_classes = 2
        c, h, sigma_data = 3, 128, 0.5881

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    # This took very long for celebA; computed once and put above, hardcoded
    # x, _ = next(iter(DataLoader(train_dataset, batch_size=10000, shuffle=True)))
    # _, c, h, w = x.size()
    # assert h == w
    # sigma_data = x.std()
    # print(c, h, sigma_data)

    return train_dataset, valid_dataset, DataInfo(c, h, num_classes, sigma_data)


def celebA_collate(batch):
    images, attrs = zip(*batch)
    images = default_collate(images)  # type: ignore
    attrs = default_collate(attrs)  # type: ignore
    last_attr = attrs[:, -1]
    return images, last_attr


def make_dataloaders(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    num_classes: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoaders:

    assert num_classes in [2, 10]

    if num_classes == 2:
        collate_fn = celebA_collate
    else:
        collate_fn = default_collate

    kwargs = {
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "persistent_workers": (num_workers > 0),
        "pin_memory": pin_memory,
    }

    return DataLoaders(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs),
        valid=DataLoader(valid_dataset, batch_size=2 * batch_size, **kwargs),
    )
