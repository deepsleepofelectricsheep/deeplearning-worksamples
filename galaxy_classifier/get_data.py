# ML, DL and vision libraries
import torch
from torch.utils import data
from torchvision import transforms
from sklearn import model_selection

# Standard imports
import numpy as np
import os
from os import path
from PIL import Image

# Required imports
from astroNN import datasets


def generate_and_save_splits(
    test_size: float = 0.1,
    val_size: float = 0.1,
    root: str = "data"    
) -> None:
    """Download the Galaxy10 dataset, create train, val
    and test splits, and store the data in the provided
    root directory. The train, val and test sets will be 
    stored in separate sub directories, and each image 
    will be stored as a separate file to enable lazy
    loading during model training. 

    Args:
        test_size: 
            The fraction of the total dataset to be used 
            for model testing.
        val_size: 
            The fraction of the total dataset tobe used 
            for model validation.
        root:
            The root directory where the data should be 
            stored.
    """

    combined_size = test_size + val_size

    # The following line will load the Galaxy10 dataset. 
    # If the dataset is not already on the disk, then it
    # will be downloaded.
    images, labels = datasets.load_galaxy10()

    # Perform train-validation-test splitting. The
    # manually-defined random state is designed to make
    # the results of this function reproducible.
    random_state = 42
    train_idx, temp_idx = model_selection.train_test_split(
        np.arange(len(labels)),
        test_size=combined_size,
        stratify=labels,
        random_state=random_state
    )
    val_idx, test_idx = model_selection.train_test_split(
        temp_idx,
        test_size=test_size/combined_size,
        stratify=labels[temp_idx],
        random_state=random_state
    )

    subdirs = ["/train/", "/test/", "/val/"]
    for subdir in subdirs:
        if not path.exists(f"{root}{subdir}/images"):
            os.makedirs(f"{root}{subdir}/images")
        if not path.exists(f"{root}{subdir}/labels"):
            os.makedirs(f"{root}{subdir}/labels")

    for idx, subdir in zip([train_idx, test_idx, val_idx], subdirs):
        for i, id in enumerate(idx):
            image = Image.fromarray(images[id])
            label = labels[id]
            
            dir = f"{root}{subdir}"
            image.save(os.path.join(f"{dir}images/", f"image_{i:05d}.png"))
            with open(os.path.join(f"{dir}labels/", f"label_{i:05d}.txt"), "w") as f:
                f.write(str(label))


class GalaxyDataset(data.Dataset):
    """PyTorch Dataset object that transforms and loads the 
    data.

    Attributes:
        image_dir:
            The sub directory where to find the images.
        label_dir:
            The sub directory where to find the labels.
        indices:
            The list of indices for the images and labels
            in the subdirectory.
        transform: 
            The transformation composer instance to apply
            to the images. 
    """

    def __init__(
        self, 
        dir: str = "data/train",
        transform: transforms.Compose | None = None,
        debug: bool = True,
    ) -> None:
        """Initializes the instance based on provided data
        
        Args:
            dir:
                The directory where the Galaxy10 image data can be 
                found. Provide either train, val, or test directory.
            transform:
                An instance of the torchvision Compose class, used 
                to transform the image data.
            debug:
                Boolean flag to determine whether to subset data for
                debugging. 
        """
        
        self.image_dir = f"{dir}/images"
        self.label_dir = f"{dir}/labels"
        self.indices = sorted([
            f.split("_")[1].split(".")[0]
            for f in os.listdir(self.image_dir)
        ])
    
        if debug == True:
            self.indices = self.indices[:32]

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224), transforms.ToTensor()
            ])

    def __getitem__(
            self, 
            idx: int
    ) -> dict[str, torch.Tensor | np.uint8]:
        """Returns a sample at the provided index, after applying 
        the specified transformation to the image.
        
        Args:
            idx: Index of sample to return.

        Returns:
            A tuple of a tensor and a numpy unsigned integer,
            representing the image and the associated label.
        """
        image_file = path.join(
            self.image_dir, f"image_{self.indices[idx]}.png"
        )
        image = Image.open(image_file)
        image = self.transform(image)

        label_file = path.join(
            self.label_dir, f"label_{self.indices[idx]}.txt"
        )
        label = int(open(label_file).read())

        return {"image": image, "label": label}
    
    def __len__(self) -> int:
        """Returns the number of elements in the Dataset."""
        return len(self.indices)


def load_data(
    dir: str = "data",
    batch_size: int = 16,
    image_transforms: dict[str, transforms.Compose] = None,
    debug: bool = True,
) -> dict[str, data.DataLoader]:
    """Lazy loads train, val and test splits into PyTorch DataLoaders
    of specified batch size.
    
    Args:
        splits: 
            A dictionary mapping the keys "train", "val" and
            "test" to the corresponding data splits. Each row
            is represented as a tuple of numpy ndarrays.
        batch_size: 
            The number of samples to be loaded per batch.
        image_transforms:
            A dictionary mapping the keys "train", "val" and
            "test" to the corresponding image transformations. 
        debug:
            Boolean flag to determine whether to subset data for
            debugging. 

    Returns:
        A dictionary mapping the keys "train", "val" and "test"
        to the corresponding DataLoaders.   
    """

    if image_transforms == None:
        image_transforms = {
            "train": None, 
            "val": None, 
            "test": None
        }

    data_loaders = {}
    splits = ["test", "train", "val"]
    for split in splits:
        dataset = GalaxyDataset(
            f"{dir}/{split}",
            image_transforms[split],
            debug
        )
        shuffle = True if split == "train" else False
        data_loader = data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
        data_loaders[split] = data_loader
    return data_loaders