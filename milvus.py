import json
from pymilvus import MilvusClient
from pymilvus.client.types import DataType
from pymilvus.orm.schema import CollectionSchema, FieldSchema
import torch 

json_file_path = r"test_embeddings_and_similarity.json"
output_json_file_path = "milvus_search_data_test.json"

# --- 1. Connettiti al tuo server Milvus ---
client = MilvusClient(uri="http://localhost:19530")

# --- 2. Definisci lo schema della collezione  ---
COLLECTION_NAME = "my_image_embeddings_for_text_query" 
VECTOR_DIMENSION = 1024

# Definisci i campi per la collezione delle immagini: solo ID e image_embedding
id_field = FieldSchema(
    name="id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=False
)
image_embedding_field = FieldSchema(
    name="image_embedding", # Questo campo conterrà gli embedding delle immagini
    dtype=DataType.FLOAT_VECTOR,
    dim=VECTOR_DIMENSION
)

# Raggruppa i campi in una CollectionSchema per la collezione delle immagini
schema = CollectionSchema(
    fields=[id_field, image_embedding_field],
    description="Collezione di embedding di immagini, interrogabile con embedding di testo",
    enable_dynamic_field=False
)

# --- 3. Prepara gli index_params per il campo dell'immagine ---
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="image_embedding", # L'indice è sul campo dell'immagine
    metric_type="IP",          # Metrica Inner Product 
    index_type="HNSW",
    params={"M": 8, "efConstruction": 64}
)


# --- 4. Gestione della Collezione (Drop se esiste, poi Crea) ---
if client.has_collection(collection_name=COLLECTION_NAME):
    print(f"La collezione '{COLLECTION_NAME}' esiste già. Eliminazione in corso...")
    client.drop_collection(collection_name=COLLECTION_NAME)
    print(f"Collezione '{COLLECTION_NAME}' eliminata.")

print(f"Creazione della collezione '{COLLECTION_NAME}' con dimensione {VECTOR_DIMENSION}...")
client.create_collection(
    collection_name=COLLECTION_NAME,
    schema=schema, 
)
print(f"Collezione '{COLLECTION_NAME}' creata con successo!")

print(f"Applicazione dell'indice al campo 'image_embedding' nella collezione '{COLLECTION_NAME}'...")
client.create_index(
    collection_name=COLLECTION_NAME,
    index_params=index_params
)
print(f"Indice per 'image_embedding' creato con successo!")

# --- 6. Carica e Prepara i Dati dal JSON ---
try:
    with open(json_file_path, 'r', encoding='utf-8') as f:
        full_json_data = json.load(f) # Carica tutti i dati JSON qui
except FileNotFoundError:
    print(f"Errore: Il file JSON non trovato al percorso: {json_file_path}")
    exit()
except json.JSONDecodeError:
    print(f"Errore: Impossibile decodificare il file JSON. Controlla la sua formattazione.")
    exit()

# Prepara i dati per l'inserimento nella collezione Milvus 
entities_to_insert_into_milvus = []
for item in full_json_data:
    if "Row_ID" in item and "image_embedding" in item:
        if len(item["image_embedding"]) == VECTOR_DIMENSION:
            entities_to_insert_into_milvus.append({
                "id": item["Row_ID"],
                "image_embedding": item["image_embedding"]
            })
        else:
            print(f"Avviso: Embedding immagine con dimensione errata per Row_ID {item['Row_ID']}. Saltato.")
    else:
        print(f"Avviso: Campi 'Row_ID' o 'image_embedding' mancanti nell'oggetto JSON: {item}. Saltato.")

print(f"Numero di entità di immagini da inserire nella collezione: {len(entities_to_insert_into_milvus)}")

# --- 7. Inserisci i Dati nella Collezione ---
if entities_to_insert_into_milvus:
    print(f"Inserimento dei dati nella collezione '{COLLECTION_NAME}'...")
    insert_result = client.insert(collection_name=COLLECTION_NAME, data=entities_to_insert_into_milvus)
    print("Risultato inserimento:", insert_result)
    print(f"Dati inseriti con successo! Numero di entità: {len(entities_to_insert_into_milvus)}")

    print(f"Caricamento della collezione '{COLLECTION_NAME}' in memoria...")
    client.load_collection(collection_name=COLLECTION_NAME)
    print("Collezione caricata.")

    all_search_results = [] 

    if full_json_data:
        print("\nEsecuzione della ricerca cross-modale per tutti i testi...")
        for query_item in full_json_data:
            query_row_id_from_json = query_item.get("Row_ID")
            sample_text_query_vector = query_item.get("text_embedding")

            if sample_text_query_vector and len(sample_text_query_vector) == VECTOR_DIMENSION:
                print(f"Cercando immagini simili per il testo con Row_ID: {query_row_id_from_json}")
                search_results = client.search(
                    collection_name=COLLECTION_NAME,
                    data=[sample_text_query_vector], # La query è un embedding di TESTO
                    limit=5, # Restituisci i 5 risultati più simili per ogni query
                    search_field="image_embedding",  # La ricerca avviene sul campo delle IMMAGINI
                    output_fields=["id"], # Restituisci l'ID dell'immagine trovata
                    metric_type="IP"     # Metrica di similarità (Inner Product)
                )

                formatted_hits = []
                if search_results and len(search_results) > 0 and len(search_results[0]) > 0:
                    for hit in search_results[0]:
                        formatted_hits.append({
                            "found_image_id": hit['id'],
                            "similarity_distance_ip": hit['distance']
                        })
                
                all_search_results.append({
                    "query_text_row_id": query_row_id_from_json,
                    "top_similar_images": formatted_hits
                })
            else:
                print(f"Avviso: Embedding di testo mancante o di dimensione errata per Row_ID {query_row_id_from_json}. Saltato.")

        # Salva tutti i risultati in un file JSON
        try:
            with open(output_json_file_path, 'w', encoding='utf-8') as outfile:
                json.dump(all_search_results, outfile, indent=4)
            print(f"\nRisultati della ricerca salvati in: {output_json_file_path}")
        except Exception as e:
            print(f"Errore durante il salvataggio dei risultati in JSON: {e}")

    else:
        print("Nessun dato valido trovato nel file JSON per eseguire le query.")

else:
    print("Nessun dato valido trovato nel file JSON per l'inserimento di immagini.")

# --- 9. Disconnetti  ---
client.close()
print("\nScript completato.")