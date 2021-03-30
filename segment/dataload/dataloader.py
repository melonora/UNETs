import torch
from torch.utils.data import Dataset, DataLoader


class SegmentDataset(Dataset):
    """
    Synthetic MICS datasets

    Attributes
    ----------
    json_dir: str
        String indicating the path to the directory containing the json files with info regarding segmentations and the
        annotations.
    root_imdir: str
        Root directory containing the images.
    """
    def __init__(self, root_imdir, json_dir):
        self.json_dir = json_dir
        self.root_imdir = root_imdir

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
