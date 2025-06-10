import json
import os
import torch
import torch.nn.functional as F


def K_text(query_text_embedding, all_image_data, k=10, device='cpu'):
    
    query_text_embedding = query_text_embedding.to(device)
    similarities = []

    for image_id, image_embedding in all_image_data.items():
        image_embedding = image_embedding.to(device)
        sim_score = torch.dot(query_text_embedding, image_embedding).item()
        similarities.append((sim_score, image_id))

    # Ordina i risultati per punteggio di similarità in ordine decrescente
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:k]

def WAS(retrieved_items_list, k_value):

    weighted_sum_scores = 0.0
    sum_weights = 0.0

    for i, result_entry in enumerate(retrieved_items_list):
        score = result_entry['dot_product_score']
        weight = k_value - i # Assegna il peso: k, k-1, ..., 1

        if weight > 0: 
            weighted_sum_scores += (score * weight)
            sum_weights += weight

    if sum_weights > 0:
        return weighted_sum_scores / sum_weights
    return None


def WAS_eval(input_json_path, output_json_path, k_value_retrieval):

    with open(input_json_path, 'r') as f:
            data = json.load(f)

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
    individual_weighted_averages = []

    if all_text_embeddings:
        print("\n--- Avvio del Retrieval Testo-Immagine ---")
        for query_text_id, query_text_embedding in all_text_embeddings.items():
            print(f"Elaborazione query testo ID: {query_text_id}")

            # Recupera i top-k risultati per la query corrente
            top_k_retrieved_items = K_text(
                query_text_embedding,
                all_image_embeddings,
                k=k_value_retrieval,
                device=current_device
            )

            # Prepara il dizionario per i risultati di questa query
            query_specific_results = {
                "query_text_id": query_text_id,
                "retrieved_images": [] # Contiene la lista delle immagini recuperate
            }

            # Popola la lista dei risultati recuperati per questa query
            for i, (score, image_id) in enumerate(top_k_retrieved_items):
                result_entry = {
                    "image_id": image_id,
                    "dot_product_score": score
                }
                # Controlla se il risultato recuperato è l'originale quindi classifica correttamente
                if image_id == query_text_id:
                    result_entry["is_original_match"] = True
                
                query_specific_results["retrieved_images"].append(result_entry)

            # Calcola la Similarità Media Pesata (WAS) per questa query
            was_for_query = WAS(
                query_specific_results["retrieved_images"], k_value_retrieval
            )

            if was_for_query is not None:
                query_specific_results["weighted_average_similarity"] = round(was_for_query, 6)
                individual_weighted_averages.append(was_for_query)
                print(f"  WAS per query {query_text_id}: {was_for_query:.4f}")
            else:
                query_specific_results["weighted_average_similarity"] = None
                print(f"  Attenzione: Impossibile calcolare WAS per query {query_text_id} (risultati insufficienti).")

            all_retrieval_results.append(query_specific_results)
    else:
        return False

    # Calcola lo Score Complessivo di Similarità Media Pesata (Overall WAS)
    overall_weighted_average = None
    if individual_weighted_averages:
        overall_weighted_average = sum(individual_weighted_averages) / len(individual_weighted_averages)
        print(f"\n--- Score Complessivo di Similarità Media Pesata: {overall_weighted_average:.4f} ---")
        # Aggiunge lo score complessivo come ultima entry nella lista dei risultati
        all_retrieval_results.append({
            "overall_weighted_average_similarity": round(overall_weighted_average, 6)
        })
    else:
        print("\nNessun dato valido per calcolare lo score complessivo di similarità media pesata.")
        return False 
    
    with open(output_json_path, 'w') as f:
            json.dump(all_retrieval_results, f, indent=4)


script_directory = os.path.dirname(__file__)

input_embeddings_json_path = os.path.join(
        os.path.abspath(os.path.join(script_directory, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir)),
        "C:/Users/Giulio/anaconda3/envs/pythorch_env/src/ContrastiveLearning/test_embeddings_and_similarity.json"
    )
output_retrieval_json_path = os.path.join(
        script_directory,
        "image_to_text_retrieval_results_with_was.json"
    )

top_k_results = 10 

print("--- Inizio dell'elaborazione del retrieval e della valutazione WAS ---")


processing_successful = WAS_eval(
        input_embeddings_json_path,
        output_retrieval_json_path,
        top_k_results
    )

print("\nElaborazione completata")