

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from multiprocessing import freeze_support


from model import BacteriaDetector, detection_loss
from dataset import BacteriaDataset
from utils import *


#Custom collate function (handles variable number of boxes)

def collate_fn(batch):
    images = []
    targets = []
    for img, boxes in batch:
        images.append(img)
        targets.append(boxes)
    images = torch.stack(images, dim=0)
    return images, targets

#Main Training Function

def train_model():


    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    
    print("🔍 Checking dataset folders...")
    for folder in ["train/images", "train/labels", "val/images", "val/labels"]:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Missing folder: {folder}")

  
    train_dataset = BacteriaDataset("train/images/", "train/labels/")
    val_dataset = BacteriaDataset("val/images/", "val/labels/")
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")

   
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    
    model = BacteriaDetector(num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

   
    anchors = [
        [(10,13), (16,30), (33,23)],
        [(30,61), (62,45), (59,119)],
        [(116,90), (156,198), (373,326)]
    ]

    num_epochs = 50 
    patience_limit = 5 
    print("🚀 Starting training...\n")

    best_val_loss = float("inf")
    patience_counter = 0

 
    # Training Loop with Validation + Early Stopping
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        
        for batch_idx, (images, targets) in enumerate(train_loader):
            try:
                images = images.to(device)
                preds = model(images)
                loss = detection_loss(preds, targets, anchors=anchors, device=device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx}] | Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx}: {e}")
                continue 
        avg_train_loss = total_loss / max(1, len(train_loader))
        print(f"✅ Epoch [{epoch+1}/{num_epochs}] | Avg Train Loss: {avg_train_loss:.4f}")

       
        #Validation Phase
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                preds = model(images)
                loss = detection_loss(preds, targets, anchors=anchors, device=device)
                val_loss += loss.item()

        val_loss /= max(1, len(val_loader))
        print(f"📊 Validation Loss after Epoch [{epoch+1}]: {val_loss:.4f}")

        #Early Stopping Logic
        
        if val_loss < best_val_loss:
            print(f"💾 Validation improved from {best_val_loss:.4f} → {val_loss:.4f}, saving best model...")
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/best_model.pth")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement for {patience_counter}/{patience_limit} epochs.")

        if patience_counter >= patience_limit:
            print("🛑 Early stopping triggered — validation loss not improving.")
            break

        print("-" * 60)

    print("\n🎉 Training complete! Best model saved as 'weights/best_model.pth'")


#Entry Point (for macOS/Windows multiprocessing safety)

if __name__ == "__main__":
    freeze_support()
    train_model()
