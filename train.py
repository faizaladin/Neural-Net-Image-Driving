import torch
from torch.utils.data import DataLoader, random_split
from dataloader import CarlaSteeringPerTownSamplesDataset
from model import Driving
import wandb
from tqdm import tqdm
import numpy as np

# Training settings
BATCH_SIZE = 128
EPOCHS = 5000
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.init(project="carla-driving", name="driving-pilotnet")

def main():
    # Dataset and DataLoader
    from torchvision import transforms
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor()
    ])
    val_transform = transforms.ToTensor()

    # Define per-town sample counts
    samples_per_town = {
        "Town02": 737,
        "Town03": 1276,
        "Town04": 1308
    }
    full_dataset = CarlaSteeringPerTownSamplesDataset('./', samples_per_town, transform=None)
    indices = np.random.permutation(len(full_dataset))
    train_size = int(0.8 * len(full_dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    # Patch the transform for each subset
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model, loss, optimizer
    model = Driving().to(DEVICE)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for i, (images, steers) in enumerate(train_bar):
            images = images.to(DEVICE)
            steers = steers.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, steers)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
            if (i+1) % 10 == 0:
                # Log the first image in the batch
                img_np = images[0].cpu().numpy().transpose(1,2,0)
                wandb.log({
                    "train/loss": loss.item(),
                    "epoch": epoch+1,
                    "train/image": [wandb.Image(img_np, caption=f"Label: {steers[0].item():.3f}, Pred: {outputs[0].item():.3f}")],
                    "train/label": steers[0].item(),
                    "train/pred": outputs[0].item()
                })
        avg_train_loss = running_loss/len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Train Loss: {avg_train_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch+1})

        # Evaluation
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        for i, (images, steers) in enumerate(val_bar):
            images = images.to(DEVICE)
            steers = steers.to(DEVICE)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, steers)
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
                if i == 0:  # Log the first batch's first image for validation
                    img_np = images[0].cpu().numpy().transpose(1,2,0)
                    wandb.log({
                        "val/image": [wandb.Image(img_np, caption=f"Label: {steers[0].item():.3f}, Pred: {outputs[0].item():.3f}")],
                        "val/label": steers[0].item(),
                        "val/pred": outputs[0].item(),
                        "epoch": epoch+1
                    })
        avg_val_loss = val_loss/len(val_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Validation Loss: {avg_val_loss:.4f}")
        wandb.log({"val/loss": avg_val_loss, "epoch": epoch+1})

        # Log a image from the validation set every epoch
        rand_idx = np.random.randint(len(val_dataset))
        img, label = val_dataset[rand_idx]
        model.eval()
        with torch.no_grad():
            img_input = img.unsqueeze(0).to(DEVICE)
            pred = model(img_input).cpu().item()
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        wandb.log({
            "val/random_image": [wandb.Image(img_np, caption=f"Label: {label.item():.3f}, Pred: {pred:.3f}")],
            "val/random_label": label.item(),
            "val/random_pred": pred,
            "epoch": epoch+1
        })
        torch.save(model.state_dict(), 'town2_3_4.pth')
        print("Model checkpoint saved as town2_3_4.pth")
        model.train()

    # Save model
    torch.save(model.state_dict(), 'town2_3_4.pth')
    print("Training complete. Model saved as town2_3_4.pth")

if __name__ == "__main__":
    main()
