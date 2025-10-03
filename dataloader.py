import os
import csv
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

# Dataset that takes a dict of {town: num_samples} and loads the first N frames for each town
class CarlaSteeringPerTownSamplesDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, samples_per_town, transform=None):
        """
        samples_per_town: dict mapping town name to number of frames to take (first N, not random)
        """
        self.data = []
        self.transform = transform
        for town in os.listdir(data_root):
            town_path = os.path.join(data_root, town)
            labels_path = os.path.join(town_path, 'labels.csv')
            if not os.path.isdir(town_path) or not os.path.isfile(labels_path):
                continue
            frames = []
            with open(labels_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame = row['frame']
                    steer = float(row['steer'])
                    frame_path = os.path.join(town_path, f"{frame}.png")
                    if os.path.isfile(frame_path):
                        frames.append((frame_path, steer, frame))
            # Sort by frame (as int if possible, else as string)
            try:
                frames.sort(key=lambda x: int(x[2]))
            except Exception:
                frames.sort(key=lambda x: x[2])
            n = samples_per_town.get(town, 0)
            if n > 0:
                for tup in frames[:n]:
                    self.data.append((tup[0], tup[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, steer = self.data[idx]
        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(steer, dtype=torch.float32)


def load_frame_steering_tuples(data_root, max_samples=1500):
    """
    For each town folder in data_root, load (frame, steering) tuples.
    If frames < max_samples, use all. Else, randomly sample max_samples.
    Returns: dict {town_name: list of (frame_path, steering)}
    """
    town_data = {}
    for town in os.listdir(data_root):
        town_path = os.path.join(data_root, town)
        labels_path = os.path.join(town_path, 'labels.csv')
        if not os.path.isdir(town_path) or not os.path.isfile(labels_path):
            continue
        frames = []
        with open(labels_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame = row['frame']
                steer = float(row['steer'])
                frame_path = os.path.join(town_path, f"{frame}.png")
                if os.path.isfile(frame_path):
                    frames.append((frame_path, steer))
        if len(frames) > max_samples:
            frames = random.sample(frames, max_samples)
        town_data[town] = frames
    return town_data

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class CarlaSteeringDataset(Dataset):
    def __init__(self, data_root, max_samples=1500, transform=None):
        from torchvision import transforms
        self.data = []
        # If no transform is provided, use ToTensor by default
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        town_data = load_frame_steering_tuples(data_root, max_samples)
        for tuples in town_data.values():
            self.data.extend(tuples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, steer = self.data[idx]
        image = pil_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(steer, dtype=torch.float32)

# if __name__ == "__main__":
#     from torchvision import transforms
#     import matplotlib.pyplot as plt

#     # No resizing, just convert to tensor
#     transform = transforms.ToTensor()
#     dataset = CarlaSteeringDataset('/home/faizaladin/Desktop/image-based-driving', transform=transform)
#     print(f"Dataset size: {len(dataset)} samples")
#     img, steer = dataset[0]
#     print(f"Image shape: {img.shape}, Steering: {steer.item()}")
#     # Display the image
#     plt.imshow(img.permute(1, 2, 0))
#     plt.title(f"Steering: {steer.item():.3f}")
#     plt.axis('off')
#     plt.show()
