from PIL import Image
import os
from torch.utils.data import Dataset
import pandas as pd

class SubmissionDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_ext, transform=None) -> None:
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name1, img_name2 = self.data.iloc[idx, 0].split('_') 
        path_img1 = os.path.normpath(os.path.join(self.img_dir, img_name1 + f'.{self.img_ext}'))
        path_img2 = os.path.normpath(os.path.join(self.img_dir, img_name2 + f'.{self.img_ext}'))
        img1 = Image.open(path_img1).convert("RGB")
        img2 = Image.open(path_img2).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, self.data.iloc[idx, 0]
