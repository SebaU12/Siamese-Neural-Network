import pandas as pd
from PIL import Image
import os
import torch 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class WildFacesDataset(Dataset):
    def __init__(self, csv_file, img_dir, img_ext, transform=None) -> None:
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name1, img_name2 = self.data.iloc[idx, 0].split('_') 
        label = self.data.iloc[idx, 1]
        path_img1 = os.path.normpath(os.path.join(self.img_dir, img_name1 + f'.{self.img_ext}'))
        path_img2 = os.path.normpath(os.path.join(self.img_dir, img_name2 + f'.{self.img_ext}'))
        img1 = Image.open(path_img1).convert("RGB")
        img2 = Image.open(path_img2).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        '''
        if label == 'same':
            label = torch.tensor([1.0, 0.0], dtype=torch.float32)
        else:
            label = torch.tensor([0.0, 1.0], dtype=torch.float32)
        '''
        if label == 'same':
            label = torch.tensor(1.0, dtype=torch.float32)
        else:
            label = torch.tensor(0.0, dtype=torch.float32)
        return img1, img2, label

def buildTrainTestDataset(batch_size, csv_file, img_dir, img_ext, transform=None):
    data = pd.read_csv(csv_file)
    # Dividir el DataFrame en entrenamiento y prueba
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_csv = 'train_data.csv'
    test_csv = 'test_data.csv'
    train_data.to_csv(train_csv, index=False)
    test_data.to_csv(test_csv, index=False)

    train_dataset = WildFacesDataset(csv_file=train_csv, img_dir=img_dir, img_ext=img_ext, transform=transform)
    test_dataset = WildFacesDataset(csv_file=test_csv, img_dir=img_dir, img_ext=img_ext, transform=transform)
    img1, img2, _ = train_dataset[0]
    print("Entrada: ", img1.shape, img2.shape)
    print("Batch size: ", batch_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader