import torch
import torch.optim as optim
import pandas as pd
import os
import time
from torchvision import transforms
from PIL import Image
import gc # Manteniamo l'import di gc

# Assumi che Siamese_Network.py contenga le classi SiameseNetwork e Similarity_Loss_Sigmoid
from Siamese_Network import SiameseNetwork, Similarity_Loss_Sigmoid_Vectorized


def load_batch_from_dataframe(df, image_folder, row_ids, transform, device):
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
            # Usa torch.zeros con requires_grad=False per evitare problemi di grafici
            images.append(torch.zeros(3, 224, 224, device=device))
            texts.append("")
        except (OSError, IndexError) as e:
            print(f"Error processing data for Row_ID {row_id}: {e}")
            images.append(torch.zeros(3, 224, 224, device=device))
            texts.append("")
    return torch.stack(images), texts

# Funzione di valutazione adattata per elaborare una coppia alla volta per batch
def evaluate(model, df_val, image_folder, loss_fn, batch_size, device):
    model.eval() # Imposta il modello in modalità valutazione
    total_loss = 0
    total_rows = len(df_val)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Disabilita il calcolo dei gradienti per l'inferenza
    with torch.no_grad():
        for start_idx in range(0, total_rows, batch_size):
            batch_df = df_val.iloc[start_idx:min(start_idx + batch_size, total_rows)]
            current_batch_row_ids = batch_df["Row_ID"].tolist()

            batch_text_embeddings = []
            batch_image_embeddings = []

            # Ciclo sulle singole coppie immagine-testo all'interno del batch di valutazione
            for row_id in current_batch_row_ids:
                # Carica una singola immagine e il suo testo
                single_image, single_text_list = load_batch_from_dataframe(df_val, image_folder, [row_id], transform, device)
                
                # single_text_list conterrà un solo elemento, prendiamo il testo
                single_text = single_text_list[0]
                single_image = single_image.to(device) # single_image è già un tensore stackato

                # Calcola gli embedding per la singola coppia
                # model si aspetta una lista di testi per coerenza con il batching, anche se è uno solo
                text_emb, image_emb = model([single_text], single_image)
                
                batch_text_embeddings.append(text_emb)
                batch_image_embeddings.append(image_emb)
                
                # Svuota la cache CUDA e la memoria Python dopo ogni elaborazione di coppia
                # Questo è meno critico in eval con no_grad, ma mantiene coerenza con train
                gc.collect()
                torch.cuda.empty_cache()

            # Concatena gli embedding raccolti per l'intero batch di valutazione
            if len(batch_text_embeddings) > 0 and len(batch_image_embeddings) > 0:
                batch_text_embeddings_tensor = torch.cat(batch_text_embeddings, dim=0)
                batch_image_embeddings_tensor = torch.cat(batch_image_embeddings, dim=0)

                # Calcola la loss sul batch di embedding
                loss = loss_fn(batch_text_embeddings_tensor, batch_image_embeddings_tensor)
                
                total_loss += loss.item()

            # Pulizia aggiuntiva dopo ogni batch
            gc.collect()
            torch.cuda.empty_cache()


    # Calcola la loss media sui batch
    batches_in_eval = (total_rows + batch_size - 1) // batch_size
    if batches_in_eval > 0:
        avg_loss = total_loss / batches_in_eval
    else:
        avg_loss = 0.0 # Gestisci il caso di un DataFrame di validazione vuoto

    model.train() # Riporta il modello in modalità addestramento
    return avg_loss

# Funzione di addestramento con leave-one-out e early stopping (rimane invariata)
def train_leave_one_out_from_folder(model, df_train, df_val, image_folder, loss_fn, optimizer, batch_size, epochs, device, patience=5, save_folder="1_vs_all"):
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
        
        # Ciclo sui batch
        for start_idx in range(0, total_rows, batch_size):
            batch_df = df_train.iloc[start_idx:min(start_idx + batch_size, total_rows)]
            current_batch_row_ids = batch_df["Row_ID"].tolist()

            batch_text_embeddings = []
            batch_image_embeddings = []

            optimizer.zero_grad() # Azzeriamo i gradienti all'inizio di ogni batch

            # Ciclo sulle singole coppie immagine-testo all'interno del batch
            for i, row_id in enumerate(current_batch_row_ids):
                # Carica una singola immagine e il suo testo
                single_image, single_text_list = load_batch_from_dataframe(df_train, image_folder, [row_id], transform, device)
                
                # single_text_list conterrà un solo elemento, prendiamo il testo
                single_text = single_text_list[0] 
                single_image = single_image.to(device) # single_image è già un tensore stackato

                # Calcola gli embedding per la singola coppia
                # model si aspetta una lista di testi per coerenza con il batching, anche se è uno solo
                text_emb, image_emb = model([single_text], single_image) 
                
                batch_text_embeddings.append(text_emb)
                batch_image_embeddings.append(image_emb)
                
                # Svuota la cache CUDA dopo ogni elaborazione di coppia
                gc.collect()
                torch.cuda.empty_cache()

            # Concatena gli embedding raccolti per l'intero batch
            if len(batch_text_embeddings) > 0 and len(batch_image_embeddings) > 0:
                batch_text_embeddings_tensor = torch.cat(batch_text_embeddings, dim=0)
                batch_image_embeddings_tensor = torch.cat(batch_image_embeddings, dim=0)

                # Calcola la loss sul batch di embedding
                loss = loss_fn(batch_text_embeddings_tensor, batch_image_embeddings_tensor)
                
                # Accumula i gradienti e poi esegui l'optimizer.step()
                loss.backward() # I gradienti si accumulano
                total_loss += loss.item()

            batches_per_epoch = (total_rows + batch_size - 1) // batch_size
            current_batch_num = start_idx // batch_size + 1
            print(f"Epoch {epoch + 1}/{epochs}, Batch {current_batch_num}/{batches_per_epoch}, Loss: {loss.item():.4f}")
            
            optimizer.step() # Esegui un singolo passo di ottimizzazione per il batch
            
            optimizer.zero_grad() # Azzera i gradienti per il prossimo batch
            gc.collect()
            torch.cuda.empty_cache() # Svuota la cache anche dopo l'optimizer.step() per maggiore sicurezza

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Calcola la loss media solo se ci sono stati batch
        if batches_per_epoch > 0:
            avg_loss = total_loss / batches_per_epoch
        else:
            avg_loss = 0.0 # O gestire come errore
        
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
    df_val = pd.read_csv(csv_path_val)

    model = SiameseNetwork(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Assicurati che Similarity_Loss_Sigmoid sia la versione vettorizzata
    loss_fn = Similarity_Loss_Sigmoid_Vectorized(0.1)
    loss_fn = loss_fn.to(device)
    train_leave_one_out_from_folder(model, df_train, df_val, image_folder, loss_fn, optimizer, batch_size=11413, epochs=100,device=device, patience=5, save_folder=save_folder)