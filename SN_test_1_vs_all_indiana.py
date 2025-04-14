import torch
import pandas as pd
import os
from torchvision import transforms
from PIL import Image

from Siamese_Network import SiameseNetwork  

def test_model(model, df_test, image_folder, transform, device):
    correct_predictions = 0
    total_predictions = len(df_test)

    with torch.no_grad():
        for index, row in df_test.iterrows():
            image_name = row['image_name'] 
            image_path = os.path.join(image_folder, image_name)
            try:
                img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
            except FileNotFoundError:
                print(f"Warning: Image file not found: {image_path}")
                continue

            text_emb_positive, image_emb = model([row['findings']], img) 

            correct_prediction = True

            # Calcola le distanze con tutti gli altri testi
            for other_index, other_row in df_test.iterrows():
                text_emb_other, _ = model([other_row['findings']], img)

                distance = torch.norm(image_emb - text_emb_other, p=4).item()
                if other_index != index:
                    if distance < torch.norm(image_emb - text_emb_positive, p=4).item():
                        correct_prediction = False
                        break
            print(correct_prediction, index, other_index) 
            if correct_prediction:
                correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    csv_path_test = r"data\indiana\indiana\dataset_test.csv"  # Percorso al CSV di test
    image_folder_test = r"data\indiana\indiana\Images\Images"  # Percorso alla cartella delle immagini
    model_path = r"1_vs_all\indiana_model\model_epoch_9.pth"  # Percorso al modello pre-addestrato

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_test = pd.read_csv(csv_path_test)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = SiameseNetwork(device=device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_model(model, df_test, image_folder_test, transform, device)