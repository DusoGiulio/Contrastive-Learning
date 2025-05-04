import torch
import torch.optim as optim
import pandas as pd
import os
import time
from torchvision import transforms
from PIL import Image

from Siamese_Network import SiameseNetwork, Similarity_Loss_SNN

# Funzione per caricare un batch di immagini e i relativi testi dal DataFrame
def load_batch_from_dataframe(df, image_folder, row_ids, transform=None, device='cpu'):
    images = []
    texts = []
    for row_id in row_ids:
        img_name = f"image_{row_id}.png"
        img_path = os.path.join(image_folder, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            if transform:
                img = transform(img)
            images.append(img)
            text = df.loc[df['Row_ID'] == row_id, 'report'].iloc[0]
            texts.append(text)
        except FileNotFoundError:
            print(f"Warning: Image file not found: {img_path} (for Row_ID {row_id})")
            images.append(torch.zeros(3, 224, 224).to(device))
            texts.append("") # Or some placeholder
        except (OSError, IndexError) as e:
            print(f"Error processing data for Row_ID {row_id}: {e}")
            images.append(torch.zeros(3, 224, 224).to(device))
            texts.append("") # Or some placeholder
    return torch.stack(images), texts

def evaluate(model, df_val, image_folder, loss_fn, batch_size, device='cpu'):
    model.eval()
    total_loss = 0
    total_rows = len(df_val)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for start_idx in range(0, total_rows, batch_size):
            batch = df_val.iloc[start_idx:min(start_idx + batch_size, total_rows)]
            row_ids = batch["Row_ID"].tolist()
            images, texts = load_batch_from_dataframe(df_val, image_folder, row_ids, transform, device)
            images = images.to(device)

            text_emb, image_emb = model(texts, images)
            loss = loss_fn(text_emb, image_emb)
            total_loss += loss.item()

    avg_loss = total_loss / (total_rows / batch_size)
    model.train()
    return avg_loss

# Funzione di addestramento con leave-one-out e early stopping
def train_leave_one_out_from_folder(model, df_train, df_val, image_folder, loss_fn, optimizer, batch_size, epochs, patience=3, save_folder="1_vs_all", device='cpu'):
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

    total_rows = len(df_train)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        df_train = df_train.sample(frac=1, random_state=epoch).reset_index(drop=True)
        for start_idx in range(0, total_rows, batch_size):
            batch = df_train.iloc[start_idx:min(start_idx + batch_size, total_rows)]
            row_ids = batch["Row_ID"].tolist()
            images, texts = load_batch_from_dataframe(df_train, image_folder, row_ids, transform, device)
            images = images.to(device)

            optimizer.zero_grad()

            text_emb, image_emb = model(texts, images)

            loss = loss_fn(text_emb, image_emb)

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
        print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_loss:.4f}, Epoch Time: {epoch_duration:.2f} seconds")

        # Evaluation on the validation set
        val_loss = evaluate(model, df_val, image_folder, loss_fn, batch_size, device)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_folder, "best_model.pth"))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"Epochs without improvement: {epochs_without_improvement}")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        # Save the model at the end of each epoch
        torch.save(model.state_dict(), os.path.join(save_folder, f"model_epoch_{epoch}.pth"))

if __name__ == '__main__':
    csv_path_train = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/mimic_data/train.csv"
    csv_path_val = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/mimic_data/validation.csv"
    image_folder = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/mimic_data/train_mimic"
    save_folder = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/1_vs_all/mimic/modello"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_train = pd.read_csv(csv_path_train)
    df_val = pd.read_csv(csv_path_val) # Load the validation dataframe

    model = SiameseNetwork(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = Similarity_Loss_SNN(temperature=0.09).to(device)

    train_leave_one_out_from_folder(model, df_train, df_val, image_folder, loss_fn, optimizer, batch_size=128, epochs=100, patience=5, save_folder=save_folder, device=device)

