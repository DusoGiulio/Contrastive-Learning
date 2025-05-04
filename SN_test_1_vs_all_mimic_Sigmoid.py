import torch
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from torch.nn.functional import normalize

from Siamese_Network import SiameseNetwork

def test_model(model, df_test, image_folder, transform, device):
    correct_predictions = 0
    total_predictions = len(df_test)

    with torch.no_grad():
        for index, row in df_test.iterrows():
            image_path = os.path.join(image_folder, f"image_{index}.png")
            try:
                img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
            except FileNotFoundError:
                print(f"Warning: Image file not found: {image_path}")
                continue

            text_emb_positive, image_emb = model([row['report']], img)
            image_emb = normalize(image_emb, p=2, dim=1)
            text_emb_positive = normalize(text_emb_positive, p=2, dim=1)

            similarity_positive = torch.dot(image_emb.squeeze(0), text_emb_positive.squeeze(0)).item()

            correct_prediction = True

            for other_index, other_row in df_test.iterrows():
                if other_index == index:
                    continue

                text_emb_other, _ = model([other_row['report']], img)
                text_emb_other = normalize(text_emb_other, p=2, dim=1)
                similarity_other = torch.dot(image_emb.squeeze(0), text_emb_other.squeeze(0)).item()

                if similarity_other < similarity_positive:
                    correct_prediction = False
                    break

            print(correct_prediction, index, other_index)
            if correct_prediction:
                correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy:.4f}")
if __name__ == '__main__':
    csv_path_test = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/mimic_data/test.csv"
    image_folder_test = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/mimic_data/test_mimic"
    model_path = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/1_vs_all/mimic/modello/best_model.pth"

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