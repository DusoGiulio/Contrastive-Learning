import json
import torch
import torch.nn.functional as F

def retrieve_top_k_texts(query_text, all_image_data, k=10, device='cpu'):
    
    query_text = query_text.to(device)
    similarities = []

    for image_id, image_embedding in all_image_data.items():
        image_embedding = image_embedding.to(device) 
        
        sim_score = torch.dot(query_text, image_embedding).item()
        
        similarities.append((sim_score, image_id))
    
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    return similarities[:k]

if __name__ == '__main__':
    json_file_path = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/test_embeddings_and_similarity.json"
    k_value = 10 

    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Errore: File JSON non trovato a {json_file_path}")
        exit()
    except json.JSONDecodeError:
        print(f"Errore: Impossibile decodificare il JSON da {json_file_path}")
        exit()

    if not data:
        print("Nessun dato trovato nel file JSON.")
        exit()

    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizzo del device: {current_device}")

    all_image_embeddings = {}
    all_text_embeddings = {}
    
    for entry in data:
        row_id = entry['Row_ID']

        img_emb = F.normalize(torch.tensor(entry['image_embedding']).float(), p=2, dim=0).to(current_device)
        txt_emb = F.normalize(torch.tensor(entry['text_embedding']).float(), p=2, dim=0).to(current_device)
        
        all_image_embeddings[row_id] = img_emb
        all_text_embeddings[row_id] = txt_emb

    all_retrieval_results = []

    if all_text_embeddings:
        
        for query_text_id, query_text_embedding in all_text_embeddings.items():
            print(f"Elaborazione Immagine Query Riga: {query_text_id}")
            
            top_k_texts = retrieve_top_k_texts(
                query_text_embedding,
                all_image_embeddings,
                k=k_value,
                device=current_device
            )
            
            image_query_results = {
                "query_text_id": query_text_id,
                "retrieved_texts": []
            }

            for i, (score, text_id) in enumerate(top_k_texts):
                result_entry = {
                    "image_id": text_id,
                    "dot_product_score": score
                }

                if text_id == query_text_id:
                    result_entry["is_original_match"] = True
                image_query_results["retrieved_texts"].append(result_entry)

            all_retrieval_results.append(image_query_results)


        output_results_json_path = "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/image_to_text_retrieval_results.json"
        with open(output_results_json_path, 'w') as f:
            json.dump(all_retrieval_results, f, indent=4) 
        
        print(f"\n--- Risultati salvati in: {output_results_json_path} ---")
