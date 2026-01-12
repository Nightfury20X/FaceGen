from datasets.cuhk_dataset import match_pairs, FaceSketchDataset
from sklearn.model_selection import train_test_split
from models.model import UNetGenerator, PatchGANDiscriminator
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import json
import torch
import warnings
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore", category=FutureWarning)

sketch_dir = "datasets/data/sketches"
photo_dir = "datasets/data/photos"


matched_pairs = match_pairs(sketch_dir, photo_dir)


train_pairs, test_pairs = train_test_split(matched_pairs, test_size=10, random_state=42)


base_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


train_ds = FaceSketchDataset(train_pairs, sketch_dir, photo_dir, base_transform, augment=True)
test_ds = FaceSketchDataset(test_pairs, sketch_dir, photo_dir, base_transform, augment=False)
print(f"Train images after augmentation: {len(train_ds)}")
print(f"Test images (no augmentation): {len(test_ds)}")
train_loader = DataLoader(train_ds, batch_size=10, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=10, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = UNetGenerator().to(device)
disc = PatchGANDiscriminator().to(device)
def lambda_rule(epoch):
    return 1.0 - max(0, epoch - 100) / 150


opt_gen = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))


scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_gen, lr_lambda=lambda_rule)
scheduler_D = torch.optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda_rule)


def d_loss_fn(real_out, fake_out):
    return 0.5 * torch.mean((real_out - 1) ** 2) + 0.5 * torch.mean(fake_out ** 2)

def g_loss_fn(fake_out):
    return 0.5 * torch.mean((fake_out - 1) ** 2)
criterion_L1 = nn.L1Loss()



ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

# parameters
LAMBDA_L1 = 100
NUM_EPOCHS = 250

scaler = GradScaler()

train_metrics = {
    "L1": [],
    "PSNR": [],
    "SSIM": [],
    "D_loss": [],
    "G_loss": []
}

for epoch in range(NUM_EPOCHS):
    gen.train()
    disc.train()

    epoch_L1, epoch_psnr, epoch_ssim = 0.0, 0.0, 0.0
    epoch_d_loss, epoch_g_loss = 0.0, 0.0
    num_batches = 0

    loop = tqdm(train_loader, leave=True)
    for batch_idx, (sketches, photos) in enumerate(loop):
        sketches, photos = sketches.to(device), photos.to(device)
        num_batches += 1

        
        opt_disc.zero_grad()
        with autocast():
            fake_photos = gen(sketches)
            D_real = disc(sketches, photos)
            D_fake = disc(sketches, fake_photos.detach())
            loss_D = d_loss_fn(D_real, D_fake)
        scaler.scale(loss_D).backward()
        scaler.step(opt_disc)
        scaler.update()

       
        opt_gen.zero_grad()
        with autocast():
            fake_photos = gen(sketches)
            D_fake = disc(sketches, fake_photos)
            loss_G_GAN = g_loss_fn(D_fake)
            loss_G_L1 = criterion_L1(fake_photos, photos) * LAMBDA_L1

            loss_G = loss_G_GAN + loss_G_L1

        scaler.scale(loss_G).backward()
        scaler.step(opt_gen)
        scaler.update()

        
        with torch.no_grad():
            epoch_L1 += F.l1_loss(fake_photos, photos).item()
            epoch_psnr += psnr(fake_photos, photos).item()
            epoch_ssim += ssim(fake_photos, photos).item()
            epoch_d_loss += loss_D.item()  
            epoch_g_loss += loss_G.item()  

        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

 
    train_metrics["L1"].append(epoch_L1 / num_batches)
    train_metrics["PSNR"].append(epoch_psnr / num_batches)
    train_metrics["SSIM"].append(epoch_ssim / num_batches)
    train_metrics["D_loss"].append(epoch_d_loss / num_batches) 
    train_metrics["G_loss"].append(epoch_g_loss / num_batches) 

    scheduler_G.step()
    scheduler_D.step()

torch.save(gen.state_dict(), "outputs/gen_epoch_250.pth")

with open("train_metrics.json", "w") as f:
    json.dump(train_metrics, f)