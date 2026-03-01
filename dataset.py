import os
import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ColorizationDataset(Dataset):
    def __init__(self, img_dir, file_names, size=(224, 224), is_train=True):
        self.img_dir = img_dir
        self.file_names = file_names
        # We assume the color photos exist. We derive the L channel directly from RGB -> LAB.
        # This is more robust as it guarantees perfect alignment during random data augmentations.
        self.color_path = os.path.join(img_dir, 'color')
        if not os.path.exists(self.color_path):
            # Fallback for Kaggle format cases
            if os.path.exists(os.path.join(img_dir, 'landscape Images', 'color')):
                self.color_path = os.path.join(img_dir, 'landscape Images', 'color')
            elif os.path.exists(os.path.join(img_dir, 'landscape Images', 'Color')):
                self.color_path = os.path.join(img_dir, 'landscape Images', 'Color')
            elif os.path.exists(os.path.join(img_dir, 'Color')):
                self.color_path = os.path.join(img_dir, 'Color')
            else:
                self.color_path = img_dir
            
        self.size = size
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
            ])
            
    def __len__(self):
        return len(self.file_names)
        
    def __getitem__(self, idx):
        img_name = self.file_names[idx]
        
        # Read color image
        color_img_path = os.path.join(self.color_path, img_name)
        img_color = Image.open(color_img_path).convert("RGB")
        
        # Apply transformation (resize, random flip)
        img_color = self.transform(img_color)
        
        img_color_np = np.array(img_color) / 255.0
        
        # Convert to LAB
        img_color_lab = rgb2lab(img_color_np)
        
        # Normalize LAB
        # L channel is 0 to 100, a and b are approx -128 to 127
        lab_tensor = torch.tensor(img_color_lab).float()
        lab_tensor = (lab_tensor + torch.tensor([0.0, 128.0, 128.0])) / torch.tensor([100.0, 255.0, 255.0])
        lab_tensor = lab_tensor.permute(2, 0, 1)

        # L channel is image (grayscale), ab channels are labels
        image_l = lab_tensor[:1, :, :]
        label_ab = lab_tensor[1:, :, :]

        return image_l, label_ab, img_name
