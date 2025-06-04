import os
from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir                              # here I can choose both the georef or the images directory
        self.mask_dir = mask_dir                                # these are the labels of the data
        self.transform = transform                              # the transformation for the data augmentation, for example
        self.image_filenames = sorted(f for f in os.listdir(image_dir) if f.startswith("tile"))    # the file names in the directory

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        # these return strings of the path and the name of each image
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, os.path.splitext(self.image_filenames[idx])[0] + '.png')  # assume same name between images and masks

        # we need a 3 channels input data, that's why we convert the images into RGB pictures
        image = Image.open(image_path).convert("RGB")   # converting the images in RGB pictures
        mask = Image.open(mask_path).convert("L")       # grayscale mask


        # This is mandatory to use the correct format of the files: tensors
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask