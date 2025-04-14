import torch
import torch.optim as optim
import pandas as pd
import os
import time
from torchvision import transforms
from PIL import Image

from Siamese_Network import SiameseNetwork, Similarity_Loss_1_vs_all  

# Funzione per caricare un batch di immagini dalla cartella
def load_images_from_folder(folder_path, start_idx, batch_size, transform=None):
    images = []
    for i in range(start_idx, start_idx + batch_size):
        img_name = f"image_{i}.png"
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
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

# Funzione di addestramento con leave-one-out 
def train_leave_one_out_from_folder(model, df_train, image_folder, loss_fn, optimizer, batch_size, epochs, stop_row, save_folder="1_vs_all", device='cpu'):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model.to(device)
    model.train()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    total_rows = stop_row
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0

        for start_idx in range(0, total_rows, batch_size):
            if start_idx + batch_size >= stop_row:
                print(f"Stopping training at row {stop_row}")
                torch.save(model.state_dict(), os.path.join(save_folder, f"model_epoch_{epoch}_row_{start_idx}.pth"))
                break

            batch = df_train.iloc[start_idx:min(start_idx + batch_size, total_rows)]
            texts = batch["report"].tolist()
            images = load_images_from_folder(image_folder, start_idx, len(batch), transform).to(device)
            text_input = texts

            optimizer.zero_grad()

            text_emb, image_emb = model(text_input, images)

            positive_pairs = [(i, i - start_idx) for i in range(start_idx, start_idx + len(batch))]
            loss = loss_fn(text_emb, image_emb, positive_pairs)

            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            total_loss += loss.item()

            batches_per_epoch = (total_rows + batch_size - 1) // batch_size
            current_batch = start_idx // batch_size + 1

            print(f"Epoch {epoch + 1}/{epochs}, Batch {current_batch}/{batches_per_epoch}, Loss: {loss.item():.4f}")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = total_loss / (total_rows / batch_size)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}, Epoch Time: {epoch_duration:.2f} seconds")

        torch.save(model.state_dict(), os.path.join(save_folder, f"model_epoch_{epoch}.pth"))

if __name__ == '__main__':
    csv_path = r"data\data_preprocessing\bootstrap_train\reports.csv"
    image_folder = r"train_mimic"
    save_folder = "1_vs_all"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_train = pd.read_csv(csv_path)

    model = SiameseNetwork(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = Similarity_Loss_1_vs_all(temperature=0.1).to(device)

    train_leave_one_out_from_folder(model, df_train, image_folder, loss_fn, optimizer, batch_size=100, epochs=10, stop_row=12681, save_folder=save_folder, device=device)