import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import json
from datasets.cuhk_dataset import match_pairs, FaceSketchDataset
from models.model import UNetGenerator
from sklearn.model_selection import train_test_split
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sketch_dir = "datasets/data/sketches"
photo_dir = "datasets/data/photos"


matched_pairs = match_pairs(sketch_dir, photo_dir)


train_pairs, test_pairs = train_test_split(matched_pairs, test_size=10, random_state=42)


base_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


test_ds = FaceSketchDataset(test_pairs, sketch_dir, photo_dir, base_transform, augment=False)
test_loader = DataLoader(test_ds, batch_size=10, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

def denormalize(tensor):
    return tensor * 0.5 + 0.5

gen = UNetGenerator().to(device)
gen.load_state_dict(torch.load("outputs/gen_epoch_250.pth"))
gen.eval()


with torch.no_grad():
 
    sketches, photos = next(iter(test_loader))
    sketches, photos = sketches.to(device), photos.to(device)
    fake_photos = gen(sketches)


    test_mae = F.l1_loss(fake_photos, photos).item()
    test_ssim = ssim(fake_photos, photos).item()
    test_psnr = psnr(fake_photos, photos).item()

    print(f"Test Results: MAE={test_mae:.4f}, SSIM={test_ssim:.4f}, PSNR={test_psnr:.4f}")

 
    sketches = denormalize(sketches).cpu()
    fake_photos = denormalize(fake_photos).cpu()
    real_photos = denormalize(photos).cpu()


    comparison = []
    for i in range(10):  
        comparison.extend([sketches[i], fake_photos[i], real_photos[i]])
    grid = make_grid(comparison, nrow=3, padding=10, pad_value=1)


    plt.figure(figsize=(20, 40))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Sketch -> Generated Photo -> Real Photo", fontsize=20)
    plt.axis('off')
    plt.show()

with open("train_metrics.json", "r") as f:
    train_metrics = json.load(f)
epochs = range(1, len(train_metrics["L1"]) + 1)


plt.figure(figsize=(12, 4))

# L1 Loss
plt.subplot(1, 3, 1)
plt.plot(epochs, train_metrics["L1"], label="L1 Loss", color='red')
plt.xlabel("Epoch")
plt.ylabel("L1 Loss")
plt.title("L1 Loss over Epochs")
plt.legend()
plt.grid(True)

# PSNR
plt.subplot(1, 3, 2)
plt.plot(epochs, train_metrics["PSNR"], label="PSNR", color='blue')
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.title("PSNR over Epochs")
plt.legend()
plt.grid(True)

# SSIM
plt.subplot(1, 3, 3)
plt.plot(epochs, train_metrics["SSIM"], label="SSIM", color='green')
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.title("SSIM over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Disc vs Gen loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_metrics["D_loss"], label="Discriminator Loss", color='red')
plt.plot(epochs, train_metrics["G_loss"], label="Generator Loss", color='blue')
plt.title("Discriminator vs. Generator Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()