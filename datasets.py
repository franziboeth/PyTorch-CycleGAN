import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        if isinstance(transforms_, list):
            self.transform = transforms.Compose(transforms_)
        else:
            self.transform = transforms_
        self.unaligned = unaligned
        self.mode = mode

        dir_A = os.path.join(root, mode, "A")
        dir_B = os.path.join(root, mode, "B")

        self.files_A = sorted(glob.glob(os.path.join(dir_A , '*.*')))
        self.files_B = sorted(glob.glob(os.path.join(dir_B, '*.*')))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert("RGB"))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB"))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert("RGB"))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))