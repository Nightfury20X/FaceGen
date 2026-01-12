import os, re
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

#Making filenames unifrom so we dont have mismathced pairs

def normalize_filename(fname):
    name = fname.lower().replace(" ", "")
    name = os.path.splitext(name)[0]
    name = re.sub(r"-sz\d+$", "", name)
    name = re.sub(r'^f2-', 'f-', name)
    name = re.sub(r'^m2-', 'm-', name)
    return name
#Pairing the correct photo to sketch

def match_pairs(sketch_dir, photo_dir):
    sketch_map = {normalize_filename(f): f for f in os.listdir(sketch_dir)}
    photo_map = {normalize_filename(f): f for f in os.listdir(photo_dir)}
    matching_keys = set(sketch_map) & set(photo_map)
    return [(sketch_map[k], photo_map[k]) for k in sorted(matching_keys)]

class FaceSketchDataset(Dataset):
    def __init__(self, pairs, sketch_dir, photo_dir, base_transform=None, augment=False):
        self.data = []
        for sk_file, ph_file in pairs:
            sketch = Image.open(os.path.join(sketch_dir, sk_file)).convert("L")
            photo = Image.open(os.path.join(photo_dir, ph_file)).convert("L")

#Data Augmentation (only train data) since we have too few images
            if augment:
                self.data.append((sketch, photo))
                self.data.append((TF.hflip(sketch), TF.hflip(photo)))
                for angle in [-5, 5]:
                    self.data.append((TF.rotate(sketch, angle), TF.rotate(photo, angle)))
                for b_factor, c_factor in [(0.9, 0.9), (1.1, 1.1)]:
                    sk_tmp = TF.adjust_brightness(sketch, b_factor)
                    sk_tmp = TF.adjust_contrast(sk_tmp, c_factor)
                    ph_tmp = TF.adjust_brightness(photo, b_factor)
                    ph_tmp = TF.adjust_contrast(ph_tmp, c_factor)
                    self.data.append((sk_tmp, ph_tmp))
        #For test data , unaugmented will be used
            else:
                self.data.append((sketch, photo))

        self.base_transform = base_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sketch_image, photo_image = self.data[idx]
        if self.base_transform:
            sketch_image = self.base_transform(sketch_image)
            photo_image = self.base_transform(photo_image)
        return sketch_image, photo_image
