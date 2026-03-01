import os
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
# Avoid depending on skimage if possible, but keeping it to match thesis details
from skimage.color import rgb2lab, lab2rgb

from model import ColorNet

def colorize_image(model, image_path, device, size=(224, 224)):
    # Read grayscale image, forcing it to RGB temporarily to extract the L channel safely
    img = Image.open(image_path).convert("RGB")
    original_size = img.size
    
    transform = transforms.Compose([
        transforms.Resize(size)
    ])
    
    img_resized = transform(img)
    img_np = np.array(img_resized) / 255.0
    
    # Convert to LAB to extract perfect Lightness channel (which is essentially Grayscale)
    img_lab = rgb2lab(img_np)
    
    # Extract L channel and normalize from [0, 100] to [0, 1]
    l_channel = img_lab[:, :, 0]
    l_channel_norm = l_channel / 100.0 
    
    # Create the tensor shape expected by the model: (Batch_Size=1, Channels=1, H=224, W=224)
    l_tensor = torch.tensor(l_channel_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Model inference
    model.eval()
    with torch.no_grad():
        ab_output = model(l_tensor)
        
    ab_output = ab_output.squeeze(0).cpu() # Shape (2, H, W)
    ab_output = ab_output.numpy() # Convert back to numpy
    
    # Denormalize AB channels output. The model learned to predict values [0, 1].
    # We must map [0,1] -> [-128, 127]
    ab_denorm = ab_output * 255.0 - 128.0
    
    # Combine back into a LAB formatted numpy array: Shape (H, W, Channels=3)
    l_denorm = l_channel_norm * 100.0
    l_denorm_ch = np.expand_dims(l_denorm, axis=-1)
    ab_denorm_ch = ab_denorm.transpose(1, 2, 0)
    
    lab_result = np.concatenate((l_denorm_ch, ab_denorm_ch), axis=2)
    
    # Convert back to RGB
    rgb_result = lab2rgb(lab_result)
    
    # Map back to [0, 255] discrete pixels
    rgb_result = (rgb_result * 255.0).clip(0, 255).astype(np.uint8)
    
    # Resize back to original image size cleanly
    final_img = Image.fromarray(rgb_result).resize(original_size, Image.Resampling.LANCZOS)
    return final_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Colorize grayscale images using a trained model checkpoints")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth checkpoint (e.g., checkpoints/colorization_model_best.pth)")
    parser.add_argument("--input", type=str, required=True, help="Path to input grayscale image or directory")
    parser.add_argument("--output", type=str, default="output", help="Path to output directory to save colorized images")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = ColorNet(use_pretrained=False).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isfile(args.input):
        out_path = os.path.join(args.output, "colorized_" + os.path.basename(args.input))
        print(f"Colorizing {args.input}...")
        res = colorize_image(model, args.input, device)
        res.save(out_path)
        print(f"Saved to {out_path}")
    elif os.path.isdir(args.input):
        for fname in os.listdir(args.input):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                in_path = os.path.join(args.input, fname)
                out_path = os.path.join(args.output, "colorized_" + fname)
                print(f"Colorizing {in_path}...")
                res = colorize_image(model, in_path, device)
                res.save(out_path)
                print(f"Saved to {out_path}")
