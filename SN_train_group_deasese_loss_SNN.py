import torch
import torch.optim as optim
import pandas as pd
import os
import time
from torchvision import transforms
from PIL import Image

from Siamese_Network import SiameseNetwork, Similarity_Loss_grouped_SNN  

def load_images_from_folder(folder_path, start_idx, batch_size, transform=None):
    images = []
    for i in range(start_idx, start_idx + batch_size):
        img_name = f"image_{i}.png"  
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            if transform:
                img = transform(img)
            images.append(img)
        except FileNotFoundError:
            print(f"Warning: Image file not found: {img_path}")
            images.append(torch.zeros(3, 224, 224))  
        except (OSError, UnboundLocalError) as e:
            print(f"Error processing image {img_path}: {e}")
            images.append(torch.zeros(3, 224, 224))  
    return torch.stack(images)

# Funzione di addestramento con loss per gruppi
def train_with_groups(
    model,
    df_train,
    df_val,
    image_folder,
    loss_fn,
    optimizer,
    batch_size,
    epochs,
    patience=5,
    save_folder="group_disease_model",
    device="cuda",
):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model.to(device)
    model.train()

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    total_rows = len(df_train)
    best_val_loss = 1000.0
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        df_train = df_train.sample(frac=1).reset_index(drop=True)

        for start_idx in range(0, total_rows, batch_size):
            batch = df_train.iloc[start_idx : min(start_idx + batch_size, total_rows)]
            texts = batch["report"].tolist()
            images = load_images_from_folder(
                image_folder, 
                start_idx, 
                len(batch), 
                transform
            ).to(device)
            text_input = texts
            groups = batch["gruppo"].tolist()

            optimizer.zero_grad()

            text_emb, image_emb = model(text_input, images)

            loss = loss_fn(text_emb, image_emb, groups)
            if callable(getattr(loss, "item", None)):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            torch.cuda.empty_cache()

            batches_per_epoch = (total_rows + batch_size - 1) // batch_size
            current_batch = start_idx // batch_size + 1
            if callable(getattr(loss, "item", None)):
                print(
                    f"Epoch {epoch + 1}/{epochs}, Batch {current_batch}/{batches_per_epoch}, Loss: {loss.item():.4f}"
                )

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = total_loss / (total_rows / batch_size)
        print(
            f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_loss:.4f}, Epoch Time: {epoch_duration:.2f} seconds"
        )

        # Calcola la loss sul set di validazione
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for start_idx in range(0, len(df_val), batch_size):
                batch_val = df_val.iloc[start_idx : min(start_idx + batch_size, len(df_val))]
                texts_val = batch_val["report"].tolist()
                images_val = load_images_from_folder(
                    image_folder, start_idx, len(batch_val), transform
                ).to(device)
                groups_val = batch_val["gruppo"].tolist()
                text_input_val = texts_val
                text_emb_val, image_emb_val = model(text_input_val, images_val)
                val_loss = loss_fn(text_emb_val, image_emb_val, groups_val)
                total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / (len(df_val) / batch_size)
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}")
            model.train()
            print(avg_val_loss,best_val_loss)
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(save_folder, "best_model.pth"))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs. Best Validation Loss: {best_val_loss:.4f}")
                    break

            torch.save(model.state_dict(), os.path.join(save_folder, f"model_epoch_{epoch}.pth"))

if __name__ == "__main__":
    csv_path_train = "gruop_deasease/splitted_dataset/train.csv"
    csv_path_val = "gruop_deasease/splitted_dataset/validation.csv"
    image_folder = "mimic_data/train_mimic"
    save_folder = "gruop_deasease/group_disease_model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_train = pd.read_csv(csv_path_train)
    df_val = pd.read_csv(csv_path_val)

    model = SiameseNetwork(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = Similarity_Loss_grouped_SNN(temperature=0.1).to(device)

    train_with_groups(
        model,
        df_train,
        df_val,
        image_folder,
        loss_fn,
        optimizer,
        batch_size=128,
        epochs=100,
        patience=5,
        save_folder=save_folder,
        device=device,
    )
