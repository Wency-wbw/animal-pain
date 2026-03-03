
import os, pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, metadata_csv, images_root, split="train", image_size=224):
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df["split"]==split].reset_index(drop=True)
        self.images_root = images_root
        self.t = transforms.Compose([transforms.Resize((image_size,image_size)), transforms.ToTensor()])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.images_root, r["path"])).convert("RGB")
        return self.t(img), int(r["label"]), r["id"]
