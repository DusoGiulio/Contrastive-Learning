import torch
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
from torch.nn.functional import normalize
import json 

from Siamese_Network import SiameseNetwork

def test_model(model, df_test, image_folder, transform, device, output_json_path="test_results.json"):
    model.eval()  
    results = [] 
    
    with torch.no_grad():
        for index, row in df_test.iterrows():
            img_name = f"image_{index}.png" 
            image_path = os.path.join(image_folder, img_name)
            
            try:
                img = Image.open(image_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
            except FileNotFoundError:
                print('Image not Found')
                continue
            except (OSError, IndexError) as e:
                continue

            text_report = row['report']

            # Creo i due embedding
            text_emb, image_emb = model([text_report], img_tensor) 
            
            # Normalizzo
            image_emb_normalized = normalize(image_emb, p=2, dim=1)
            text_emb_normalized = normalize(text_emb, p=2, dim=1)

            # Distanza con moltiplicazione vettoriale
            similarity_positive = torch.dot(image_emb_normalized.squeeze(0), text_emb_normalized.squeeze(0)).item()

            # Store results for the current pair
            results.append({
                "Row_ID": index,
                "image_embedding": image_emb_normalized.squeeze(0).cpu().numpy().tolist(), # Convert to list for JSON
                "text_embedding": text_emb_normalized.squeeze(0).cpu().numpy().tolist(), # Convert to list for JSON
                "positive_cosine_similarity": similarity_positive
            })
            
    # Save results to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4) 

    print(f"Test results saved to {output_json_path}")


if __name__ == '__main__':
    csv_path_test = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/mimic_data/test.csv"
    image_folder_test = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/mimic_data/test_mimic"
    model_path = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/1_vs_all/mimic/modello/best_model_test.pth"
    output_json_file = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/test_embeddings_and_similarity_t10.json" 

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

    test_model(model, df_test, image_folder_test, transform, device, output_json_path=output_json_file)