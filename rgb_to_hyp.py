import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Dummy model: Upsample RGB to 31-band hyperspectral (for demo only)
class DummyHSINet(torch.nn.Module):
    def __init__(self, out_channels=31):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def load_rgb_image(path):
    img = cv2.imread(path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize
    return img

def save_hsi_image(output_path, hsi_data):
    # Save each band as grayscale image
    os.makedirs(output_path, exist_ok=True)
    for i in range(hsi_data.shape[2]):
        band = (hsi_data[:, :, i] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_path, f"band_{i+1:02d}.png"), band)

def convert_folder(input_folder, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DummyHSINet(out_channels=31).to(device)
    model.eval()

    # Process each subdirectory (train, valid, test)
    for subdir in ['train', 'valid', 'test']:
        subdir_path = os.path.join(input_folder, subdir)
        if not os.path.exists(subdir_path):
            print(f"Skipping {subdir} - directory not found")
            continue
            
        # Create corresponding output subdirectory
        subdir_output = os.path.join(output_folder, subdir)
        os.makedirs(subdir_output, exist_ok=True)

        # Walk through all subdirectories
        for root, _, files in os.walk(subdir_path):
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for img_file in tqdm(image_files, desc=f"Converting {subdir}"):
                img_path = os.path.join(root, img_file)
                # Preserve the directory structure in output
                rel_path = os.path.relpath(root, subdir_path)
                out_path = os.path.join(subdir_output, rel_path, os.path.splitext(img_file)[0])
                
                img = load_rgb_image(img_path)
                img_tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    hsi_tensor = model(img_tensor).squeeze(0).cpu().numpy()
                hsi_data = np.transpose(hsi_tensor, (1, 2, 0))  # [H, W, C]
                save_hsi_image(out_path, hsi_data)

# Example usage:
current_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(current_dir, "Plant_Disease_Dataset", "Plant_Disease_Dataset")
output_folder = os.path.join(current_dir, "hypimages")

print(f"Looking for data in: {input_folder}")
print(f"Output will be saved to: {output_folder}")

# Check if directories exist
for subdir in ['train', 'valid', 'test']:
    path = os.path.join(input_folder, subdir)
    print(f"Checking if {path} exists: {os.path.exists(path)}")

convert_folder(input_folder, output_folder)
