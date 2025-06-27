import json
import pandas as pd

# File paths
input_result_file = "milvus_search_data_test_t1.json"
csv_report_file = "mimic_data/test.csv"
output_file = "milvus_t1_text_upgrade.json"

# Carica i risultati delle ricerche Milvus
with open(input_result_file, 'r', encoding='utf-8') as f:
    search_results = json.load(f)

# Carica il CSV contenente i report
df_reports = pd.read_csv(csv_report_file)

# Verifica che la colonna 'report' esista
if "report" not in df_reports.columns:
    raise ValueError("La colonna 'report' non è presente in test.csv")

# Lista dei report posizionali
reports = df_reports["report"].tolist()

# Costruisci il nuovo output con i report testuali associati
enriched_results = []

for entry in search_results:
    query_index = entry["query_text_row_id"]

    # Recupera il report del testo (query)
    query_report = reports[query_index] if query_index < len(reports) else "[Indice fuori range]"

    enriched_entry = {
        "query_text_row_id": query_index,
        "query_text_report": query_report,
        "top_similar_images": []
    }

    for match in entry.get("top_similar_images", []):
        image_index = match["found_image_id"]
        similarity = match["similarity_distance_ip"]

        image_report = reports[image_index] if image_index < len(reports) else "[Indice fuori range]"

        enriched_entry["top_similar_images"].append({
            "found_image_id": image_index,
            "similarity_distance_ip": similarity,
            "associated_report": image_report
        })

    enriched_results.append(enriched_entry)

# Salva il file JSON arricchito
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(enriched_results, f, indent=4)

print(f"✅ File arricchito salvato in: {output_file}")
