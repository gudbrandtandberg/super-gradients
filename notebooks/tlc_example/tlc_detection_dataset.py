import torch
import numpy as np

from torch.utils.data import Dataset

from PIL import Image

def sample_mapper(sample: dict) -> tuple[torch.Tensor, torch.Tensor]:
    path = sample["image"]
    image = Image.open(path).resize((640, 640))
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()

    labels = []

    for box in sample["bbs"]["bb_list"]:
        labels.append([box["label"], box["x0"], box["y0"], box["x1"], box["y1"]])

    if len(labels) > 0:
        labels = np.array(labels, dtype=np.float32)
    else:
        labels = np.zeros((0, 5))

    return image_tensor, labels

class TLCDetectionDataset(Dataset):
    def __init__(self, table):
        self.table = table

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the sample at the given index
        
        :param i: index of the sample
        :return: tuple of image and targets
        """
        return sample_mapper(self.table[i])