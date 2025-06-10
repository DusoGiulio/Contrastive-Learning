import torch
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score

from Siamese_Network import SiameseNetwork, Similarity_Loss_grouped_SNN

def test_with_groups(model, df_test, image_folder, transform, device):
    model.eval()
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

            correct_text = row['report']
            correct_group = row['gruppo']

            text_emb_correct, image_emb = model([correct_text], img)
            distance_correct = torch.norm(image_emb - text_emb_correct, p=2).item()

            found_closer_different_group = False
            for other_index, other_row in df_test.iterrows():
                if other_index != index and other_row['gruppo'] != correct_group:
                    other_text = other_row['report']
                    text_emb_other, _ = model([other_text], img)
                    distance_other = torch.norm(image_emb - text_emb_other, p=2).item()
                    if distance_other < distance_correct:
                        found_closer_different_group = True
                        break

            if not found_closer_different_group:
                correct_predictions += 1
                print('Correct Prediction=',correct_predictions)
                print('Correct Group=',correct_group)
                print('Corretct Row=', row['row_report_index'])

    accuracy = correct_predictions / total_predictions
    print(f"Test Accuracy (Verification of Closest Same-Group): {accuracy:.4f}")

if __name__ == '__main__':
    csv_path_test = "gruop_deasease/splitted_dataset/test.csv"
    image_folder_test = "mimic_data/test_mimic"
    model_path = "gruop_deasease/group_disease_model/best_model_SNN_group.pth" # Assicurati che il path sia corretto

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

    test_with_groups(model, df_test, image_folder_test, transform, device)