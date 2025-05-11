from astroNN import datasets
from sklearn import model_selection
import numpy as np
from torch.utils import data
from torchvision import transforms
import torch
from PIL import Image 


def generate_train_val_test_splits(
    test_size: float = 0.15,
    val_size: float = 0.15,
    debug: bool = False,
    batch_size: int = 32
) -> dict[str, tuple[np.ndarray]]:
    """Download the Galaxy10 dataset, and create train, 
    validation and test splits for model training.

    Args:
        test_size: 
            The fraction of the total dataset to be used 
            for model testing.
        val_size: 
            The fraction of the total dataset tobe used 
            for model validation.
        debug: 
            If True only a subset of the train, val, and
            test data will be returned.
        batch_size: 
            The number of samples to be returned per train,
            val and test sets if debug is set to True.
    
    Returns:
        A dictionary mapping the keys "train", "val" and
        "test" to the corresponding data splits. Each row
        is represented as a tuple of numpy ndarrays.
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

    if debug == True:
        train_idx = train_idx[:batch_size]
        val_idx = val_idx[:batch_size]
        test_idx = test_idx[:batch_size]

    return {
        "train": [images[train_idx], labels[train_idx]],
        "val": [images[val_idx], labels[val_idx]],
        "test": [images[test_idx], labels[test_idx]]
    }


class GalaxyDataset(data.Dataset):
    """PyTorch Dataset object that transforms and loads the 
    data.

    Attributes:
        images: 
            A numpy ndarray containing galaxy images.
        labels:
            A numpy ndarray containing galaxy labels.
        transform:
            An instance of the torchvision Compose class,
            used to transform the image data.
    """

    def __init__(
        self, 
        images: np.ndarray,
        labels: np.ndarray,
        transform: transforms.Compose | None = None
    ) -> None:
        """Initializes the instance based on provided data
        
        Args:
            images: The galaxy images.
            labels: The galaxy labels.
            transform: The image transformation.
        """
        self.images = images
        self.labels = labels
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
        image = Image.fromarray(self.images[idx].astype("uint8"))
        image = self.transform(image)
        label = self.labels[idx]
        return {"image": image, "label": label}
    
    def __len__(self) -> int:
        """Returns the number of elements in the Dataset."""
        return len(self.labels)


def load_data(
    splits: dict[str, tuple[np.ndarray]],
    batch_size: int = 16,
    image_transforms: dict[str, transforms.Compose] = None
) -> dict[str, data.DataLoader]:
    """Loads train, val and test splits into PyTorch DataLoaders
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
    for split in splits:
        dataset = GalaxyDataset(
            splits[split][0], 
            splits[split][1],
            image_transforms[split]
        )
        shuffle = True if split == "train" else False
        data_loader = data.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle
        )
        data_loaders[split] = data_loader
    return data_loaders