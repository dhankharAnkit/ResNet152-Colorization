import os
import argparse
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import ColorNet
from dataset import ColorizationDataset

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps") # Mac Apple Silicon Support
    else:
        return torch.device("cpu")

def train(args):
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    device = get_device()
    logger.info(f"Using device: {device}")

    # Prepare datasets
    color_dir = os.path.join(args.data_dir, 'color')
    if not os.path.exists(color_dir):
        # Fallback for Kaggle format
        if os.path.exists(os.path.join(args.data_dir, 'landscape Images', 'color')):
            color_dir = os.path.join(args.data_dir, 'landscape Images', 'color')
        elif os.path.exists(os.path.join(args.data_dir, 'landscape Images', 'Color')):
            color_dir = os.path.join(args.data_dir, 'landscape Images', 'Color')
        elif os.path.exists(os.path.join(args.data_dir, 'Color')):
            color_dir = os.path.join(args.data_dir, 'Color')
        else:
            logger.error(f"Color directory not found. Please ensure dataset has a 'color' folder.")
            return

    all_files = [f for f in os.listdir(color_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(all_files) == 0:
        logger.error("No images found in the dataset directory.")
        return

    # Shuffle and split 80/20 train/test
    torch.manual_seed(42)
    indices = torch.randperm(len(all_files)).tolist()
    
    split_idx = int(len(all_files) * 0.8)
    train_files = [all_files[i] for i in indices[:split_idx]]
    test_files = [all_files[i] for i in indices[split_idx:]]

    logger.info(f"Loaded {len(train_files)} training images and {len(test_files)} testing images.")

    # Create DataLoaders
    # Batch size explicitly reduced for MPS memory constraints by default if running natively
    train_dataset = ColorizationDataset(args.data_dir, train_files, is_train=True)
    test_dataset = ColorizationDataset(args.data_dir, test_files, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize Model, Loss, Optimizer
    model = ColorNet(use_pretrained=args.use_pretrained).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Mixed precision supported via NVIDIA Cuda scaler, or basic autocast for MPS fallback
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Track best validation loss to save the best model
    best_val_loss = float('inf')
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels, _ in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        train_loss = train_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                if scaler:
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(test_dataset)
        logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save Best Model Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'colorization_model_best.pth')
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved new best model checkpoint to {save_path}")

    # Save Final Model Checkpoint
    final_save_path = os.path.join(args.save_dir, 'colorization_model_epoch_final.pth')
    torch.save(model.state_dict(), final_save_path)
    logger.info(f"Finished Training! Final model saved to {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-152 for Image Colorization")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to the dataset directory containing 'color' folder")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training. Default to 8 to avoid memory swap on 16GB Macs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--no_pretrained", dest='use_pretrained', action='store_false', help="Do not use ImageNet pretrained weights")
    
    args = parser.parse_args()
    train(args)
