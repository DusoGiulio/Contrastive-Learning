import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from chexnet.chexnet import DenseNet121

# RADBERT
class TextEncoder(nn.Module):
    def __init__(self, model_path="./bert/radbert_local", device='cuda'):
        super(TextEncoder, self).__init__()
        self.device = device 
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert = AutoModelForMaskedLM.from_pretrained(model_path)
#Livelli densi aggiunti
        self.mlp1 = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 1024),#dovrebbepassare da 768 a 1024
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 1024),
        )

    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = self.bert(**tokens, output_hidden_states=True)
        cls_embedding = outputs.hidden_states[-1][:, 0, :]
        return F.normalize(self.mlp2(self.mlp1(cls_embedding)), p=4, dim=1)

# Modello CheXNet per le immagini
class ImageEncoder(nn.Module):
    def __init__(self, model_path="chexnet/model.pth", device='cuda'):
        super(ImageEncoder, self).__init__()
        self.device = device 
        self.model = DenseNet121(out_size=1024)

        state_dict = torch.load(model_path, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()
        self.model.densenet121.classifier = nn.Identity()
#Livelli densi aggiunti
        self.mlp1 = nn.Sequential(
            nn.Linear(1024, 1024),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 1024),
        )

    def forward(self, image):
        image = image.to(self.device) 
        with torch.no_grad():
            features = self.model(image)
        return F.normalize(self.mlp2(self.mlp1(features)), p=4, dim=1)

# Modello Siamese completo
class SiameseNetwork(nn.Module):
    def __init__(self,device):  
        super(SiameseNetwork, self).__init__()
        self.device = device
        self.text_encoder = TextEncoder(device=self.device).to(self.device)
        self.image_encoder = ImageEncoder(device=self.device).to(self.device)

    def forward(self, text, image):
        text_embedding = self.text_encoder(text)
        image_embedding = self.image_encoder(image)
        return text_embedding, image_embedding
    

    
#Funzione di loss "leave one out"
class Similarity_Loss_1_vs_all(nn.Module):
    def __init__(self, temperature=0.05):
        super(Similarity_Loss_1_vs_all, self).__init__()
        self.temperature = temperature

    def forward(self, text_embeddings, image_embeddings):
        B = text_embeddings.shape[0]
        total_loss = 0

        for i in range(B):
            image_i = image_embeddings[i].unsqueeze(0)
            distances_image_text = torch.norm(image_i - text_embeddings, dim=1)
            similarities_image_text = torch.exp(-distances_image_text / self.temperature)

            numerator = similarities_image_text[i]

            # Calcola il denominatore: somma delle similarità tra l'immagine i-esima
            # e tutti i testi che non sono il testo i-esimo
            denominator = torch.sum(similarities_image_text) - similarities_image_text[i]

            # Evita divisioni per zero nel caso (improbabile) che tutte le similarità siano zero
            if denominator > 0:
                total_loss += -torch.log(numerator / denominator)
            else:
                # Gestisci il caso in cui il denominatore è zero (potrebbe indicare problemi con i dati)
                # Potresti restituire 0 o un valore molto grande per la loss
                total_loss += 0.0  # O un altro valore appropriato

        return total_loss / B

#Funzione di loss per gruppi 
class Similarity_Loss_group_deasease(nn.Module):
    def __init__(self, temperature=0.05):
        super(Similarity_Loss_group_deasease, self).__init__()
        self.temperature = temperature

    def forward(self, text_embeddings, image_embeddings, groups):
        B = text_embeddings.shape[0]
        total_loss = 0

        for i in range(B):
            text_i = text_embeddings[i].unsqueeze(0)  # (1, dim embedding)
            distances = torch.norm(text_i - image_embeddings, dim=1)  # (B,)
            similarities = torch.exp(-distances / self.temperature)  # (B,)

            numerator = 0
            denominator = 0

            # Numeratore Similarità con esempi dello stesso gruppo escludendo se stesso
            for j in range(B):
                if groups[i] == groups[j] :  # Aggiunta la condizione i != j
                    numerator += similarities[j]

            # Denominatore Somma delle similarità con tutti gli esempi di ALTRI gruppi
            for j in range(B):
                if groups[i] != groups[j]:  # Considera solo esempi di altri gruppi
                    denominator += similarities[j]

            # Evita la divisione per zero
            if denominator > 0 and numerator > 0:
                total_loss += -torch.log(numerator / denominator)
            #elif denominator == 0 and numerator >0: # gestione caso particolare con num >0 e den =0
            #    total_loss += 0
            #elif denominator > 0 and numerator ==0: # gestione caso particolare con den >0 e num =0
            #    total_loss += 0
            #elif denominator == 0 and numerator ==0: # gestione caso particolare con den = 0 e num =0
            #    total_loss+=0
        return total_loss / B