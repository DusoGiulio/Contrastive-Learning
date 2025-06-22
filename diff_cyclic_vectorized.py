import torch
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from Siamese_Network import SiameseNetwork, Similarity_Loss_Sigmoid_Cyclic, Similarity_Loss_Sigmoid_Vectorized 
import gc
import time

def load_batch_for_test(df, image_folder, row_ids, transform, device):
    images = []
    texts = []

    df_map = df.set_index('Row_ID')

    for row_id in row_ids:
        img_name = f"image_{row_id}.png"
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        if transform:
            img = transform(img)
        images.append(img)
        # Recupera il testo usando l'indice 'Row_ID'
        text = df_map.loc[row_id, 'report']
        texts.append(text)

    return torch.stack(images).to(device), texts

csv_path_test = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/mimic_data/train.csv"
image_folder_test = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/mimic_data/train_mimic"
model_path = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/1_vs_all/mimic/modello/best_model.pth" 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_test = pd.read_csv(csv_path_test)

# Trasformazioni per le immagini
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = SiameseNetwork(device=device).to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modello caricato da {model_path}")
except FileNotFoundError:
    print(f"Attenzione: Modello non trovato in {model_path}. Gli embedding saranno casuali.")

model.eval() 
for num in [1,10,20,40,80,160,320]:
    test_row_ids = df_test['Row_ID'].head(num).tolist() 
    images_batch, texts_batch = load_batch_for_test(df_test, image_folder_test, test_row_ids, transform, device)

    # embedding dal modello Siamese
    with torch.no_grad(): 
        text_embeddings, image_embeddings = model(texts_batch, images_batch)

    print(f"Dimensioni degli embedding di testo: {text_embeddings.shape}")
    print(f"Dimensioni degli embedding di immagine: {image_embeddings.shape}")

    temperature = 0.1 
    loss_fn_v1 = Similarity_Loss_Sigmoid_Cyclic(temperature=temperature).to(device)
    loss_fn_v2 = Similarity_Loss_Sigmoid_Vectorized (temperature=temperature).to(device)

    # Calcola la loss con la versione 1 (cicli)
    start_time_v1 = time.time()
    loss_v1 = loss_fn_v1(text_embeddings, image_embeddings)
    end_time_v1 = time.time()
    print(f"Tempo di calcolo (Cicli): {end_time_v1 - start_time_v1:.4f} secondi")

    # Calcola la loss con la versione 2 (vettorizzata)
    start_time_v2 = time.time()
    loss_v2 = loss_fn_v2(text_embeddings, image_embeddings)
    end_time_v2 = time.time()
    print(f"Tempo di calcolo (Vettorizzata): {end_time_v2 - start_time_v2:.4f} secondi")
    # Confronta i risultati
    are_losses_equal = torch.isclose(loss_v1, loss_v2, rtol=1e-6, atol=1e-12) 

    if not are_losses_equal:
        print("Loss differenti :", loss_v1-loss_v2)
    else:
        print("Loss Uguali :",loss_v1-loss_v2)
    print("____________________________________________________")
    del images_batch, texts_batch, text_embeddings, image_embeddings
    del loss_v1, loss_v2
    gc.collect()
    torch.cuda.empty_cache()
