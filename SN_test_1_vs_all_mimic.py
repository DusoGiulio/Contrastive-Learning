
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
            image_path = os.path.join(image_folder, f"image_{index}.png")
            try:
                img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
            except FileNotFoundError:
                print(f"Warning: Image file not found: {image_path}")
                continue

            text_emb_positive, image_emb = model([row['report']], img)  # Embedding del testo positivo
            
            correct_prediction = True  

            # Calcola le distanze con tutti gli altri testi
            for other_index, other_row in df_test.iterrows():
                text_emb_other, _ = model([other_row['report']], img)  

                distance = torch.norm(image_emb - text_emb_other, p=4).item()  # Distanza euclidea
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
    csv_path_test = r"data\data_preprocessing\bootstrap_test\reports.csv"
    image_folder_test = r"test_mimic"
    model_path = r"1_vs_all\model_epoch_9_row_12600.pth"

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


'''import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from Siamese_Network import SiameseNetwork  # Assicurati che model.py sia nello stesso percorso
import os
import pandas as pd

# Percorsi e parametri (Modifica se necessario)
MODEL_PATH = r"1_vs_all\model_epoch_9_row_12600.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = (224, 224)  # Dimensione delle immagini utilizzata nel modello
CSV_PATH = r"data\data_preprocessing\bootstrap_test\reports.csv" # Path del CSV

# Carica il modello (una volta all'avvio)
model = SiameseNetwork(device=DEVICE).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Trasformazioni per le immagini
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def select_image():
    global image_path, pil_image
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if image_path:
        pil_image = Image.open(image_path).convert('RGB')
        pil_image.thumbnail((200, 200))  # Ridimensiona per la visualizzazione
        tk_image = ImageTk.PhotoImage(pil_image)
        image_label.config(image=tk_image)
        image_label.image = tk_image  # Mantieni un riferimento!

def update_text_preview(event):
    selected_index = text_listbox.curselection()
    if selected_index:
        selected_index = int(selected_index[0])
        selected_text = df.iloc[selected_index]['report']
        text_preview.config(state=tk.NORMAL)
        text_preview.delete("1.0", tk.END)
        text_preview.insert(tk.END, selected_text)
        text_preview.config(state=tk.DISABLED)

def calculate_similarity():
    if not image_path or not text_listbox.curselection():
        messagebox.showerror("Error", "Please select both an image and text.")
        return

    try:
        img = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(DEVICE)
        selected_index = int(text_listbox.curselection()[0])
        text = [df.iloc[selected_index]['report']]

        with torch.no_grad():
            text_emb, image_emb = model(text, img)
            distance = torch.norm(text_emb - image_emb, p=2, dim=1).item()
            similarity = -distance

        result_label.config(text=f"Similarity: {similarity:.4f}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# --- Interfaccia Grafica ---
window = tk.Tk()
window.title("Image-Text Similarity")

# Frame per la selezione dell'immagine
image_frame = tk.Frame(window)
image_frame.pack(pady=10)

image_label = tk.Label(image_frame, text="No Image Selected", width=50, height=20)
image_label.pack()

select_image_button = tk.Button(image_frame, text="Select Image", command=select_image)
select_image_button.pack(pady=5)

# Frame per la selezione del testo dal CSV
text_frame = tk.Frame(window)
text_frame.pack(pady=10)

text_label = tk.Label(text_frame, text="Select Text from CSV:")
text_label.pack()

# Carica il DataFrame
df = pd.read_csv(CSV_PATH)
text_listbox = tk.Listbox(text_frame, width=50, height=10)
for index, row in df.iterrows():
    text_listbox.insert(tk.END, f"{index}: {row['report'][:50]}...")
text_listbox.pack(pady=5)
text_listbox.bind('<<ListboxSelect>>', update_text_preview)

text_preview_label = tk.Label(text_frame, text="Preview:")
text_preview_label.pack()
text_preview = tk.Text(text_frame, width=50, height=5, state=tk.DISABLED)
text_preview.pack()

# Pulsante per calcolare la similarità
calculate_button = tk.Button(window, text="Calculate Similarity", command=calculate_similarity)
calculate_button.pack(pady=10)

# Etichetta per mostrare il risultato
result_label = tk.Label(window, text="Similarity: ")
result_label.pack(pady=10)

# Inizializzazione delle variabili globali
image_path = None
pil_image = None

window.mainloop()

'''
